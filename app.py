from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import dlib
import cv2
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete'

# Configuration de la base de données
DATABASE = 'bank.db'

# Dossier pour stocker les images des utilisateurs
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Chemins des modèles dlib
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# Charger les modèles dlib
detector = dlib.get_frontal_face_detector()  # Détecteur de visages
sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)  # Prédicteur de points de repère faciaux
facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_PATH)  # Modèle de reconnaissance faciale

# Dictionnaire pour stocker les encodages des visages
known_face_encodings = []
known_face_names = []

# Charger les visages connus
def load_known_faces():
    global known_face_encodings, known_face_names
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    image = dlib.load_rgb_image(image_path)
                    faces = detector(image)
                    if len(faces) > 0:
                        shape = sp(image, faces[0])
                        face_encoding = np.array(facerec.compute_face_descriptor(image, shape))
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)
                        print(f"Visage chargé : {person_name}")
                except Exception as e:
                    print(f"Erreur lors du chargement de l'image {image_path} : {e}")

load_known_faces()

# Initialisation de la base de données
def init_db():
    with sqlite3.connect(DATABASE, timeout=10) as conn:
        conn.execute('PRAGMA journal_mode=WAL;')  # Activer le mode WAL pour éviter les verrous
        cursor = conn.cursor()

        # Table des utilisateurs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('ADMIN', 'USER', 'EMPLOYEE')),
                balance REAL DEFAULT 0,
                last_balance_update DATE
            )
        ''')

        # Table des transactions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id INTEGER,
                receiver_id INTEGER,
                amount REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sender_id) REFERENCES users (id),
                FOREIGN KEY (receiver_id) REFERENCES users (id)
            )
        ''')

        # Table des factures
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                amount REAL,
                description TEXT,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'PAID')),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Table des prêts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                amount REAL,
                interest_rate REAL,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'APPROVED', 'REJECTED')),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Table balance_history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('ADD', 'SUBTRACT')),
                performed_by INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                requires_approval BOOLEAN DEFAULT FALSE
            )
        ''')

        # Ajouter l'administrateur par défaut
        cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        admin = cursor.fetchone()
        if not admin:
            cursor.execute('INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, ?)',
                           ('admin', generate_password_hash('admin123'), 'Admin', 'ADMIN'))
            conn.commit()

init_db()

# Décorateur pour vérifier les rôles
def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('user_id'):
                return redirect(url_for('login'))
            with sqlite3.connect(DATABASE, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT role FROM users WHERE id = ?', (session['user_id'],))
                user_role = cursor.fetchone()[0]
                if user_role != role:
                    flash('Accès refusé.', 'error')
                    return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Fonction utilitaire pour obtenir l'utilisateur actuel
def get_current_user():
    if not session.get('user_id'):
        return None
    with sqlite3.connect(DATABASE, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
        return cursor.fetchone()

# Routes principales
@app.route('/')
def index():
    return render_template("index.html")

def is_face_already_registered(new_face_encoding):
    """
    Vérifie si le visage de l'utilisateur existe déjà dans le dataset.
    :param new_face_encoding: L'encodage du visage de l'utilisateur en cours d'inscription.
    :return: True si le visage existe déjà, False sinon.
    """
    for known_encoding in known_face_encodings:
        # Calculer la distance entre les encodages
        distance = np.linalg.norm(new_face_encoding - known_encoding)
        if distance < 0.5:  # Seuil de similarité
            return True
    return False

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        role = request.form['role']

        # Validation des entrées
        if not username or not password or not full_name or not role:
            flash('Tous les champs sont obligatoires.', 'error')
            return redirect(url_for('register'))

        with sqlite3.connect(DATABASE, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                flash('Ce nom d\'utilisateur existe déjà.', 'error')
                return redirect(url_for('register'))

        # Capturer des images pour la reconnaissance faciale
        cap = cv2.VideoCapture(0)
        img_id = 0
        new_face_encodings = []

        while img_id < 10:  # Capturer 10 images
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de capturer l'image.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_frame)

            if len(faces) == 0:
                print("Aucun visage détecté !")
            else:
                print(f"Visage détecté : {faces}")

            for face in faces:
                shape = sp(rgb_frame, face)
                face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))
                new_face_encodings.append(face_encoding)

                # Vérifier si le visage existe déjà dans le dataset
                if is_face_already_registered(face_encoding):
                    flash('Ce visage est déjà enregistré. Veuillez contacter l\'administration.', 'error')
                    cap.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('register'))

                # Sauvegarder l'image dans le dossier de l'utilisateur
                user_folder = os.path.join(DATASET_DIR, username)
                if not os.path.exists(user_folder):
                    os.makedirs(user_folder)
                face_path = os.path.join(user_folder, f"{img_id}.jpg")
                cv2.imwrite(face_path, frame)
                img_id += 1

            if cv2.waitKey(1) == 13 or img_id >= 10:  # Appuyer sur Entrée pour arrêter
                break

        cap.release()
        cv2.destroyAllWindows()

        # Si aucun visage n'a été détecté
        if img_id == 0:
            flash('Aucun visage détecté. Veuillez réessayer.', 'error')
            return redirect(url_for('register'))

        # Ajouter l'utilisateur à la base de données
        hashed_password = generate_password_hash(password)
        with sqlite3.connect(DATABASE, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, ?)',
                           (username, hashed_password, full_name, role))
            conn.commit()

        # Recharger les visages connus
        load_known_faces()

        flash('Inscription réussie !', 'success')
        return redirect(url_for('login'))

    return render_template("register.html")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Connexion classique (username/mot de passe)
        if 'login_with_credentials' in request.form:
            username = request.form.get('username')
            password = request.form.get('password')

            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
                user = cursor.fetchone()

                if user and check_password_hash(user[2], password):
                    session['user_id'] = user[0]
                    session['role'] = user[4]
                    flash('Connexion réussie !', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Nom d\'utilisateur ou mot de passe incorrect.', 'error')
                    return redirect(url_for('login'))

        # Connexion par reconnaissance faciale
        elif 'login_with_face' in request.form:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                flash("Erreur: Impossible d'ouvrir la caméra.", "error")
                return redirect(url_for('login'))

            user_found = False
            name = "Inconnu"
            start_time = datetime.now()

            try:
                while (datetime.now() - start_time).total_seconds() < 10:  # Limite à 10 secondes
                    ret, frame = cap.read()
                    if not ret:
                        flash("Erreur: Impossible de capturer l'image.", "error")
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = detector(rgb_frame)

                    for face in faces:
                        shape = sp(rgb_frame, face)
                        face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

                        # Comparer avec les visages connus
                        distances = [np.linalg.norm(face_encoding - known_face) for known_face in known_face_encodings]
                        min_distance = min(distances)

                        # Vérifier si la distance minimale est inférieure au seuil
                        if min_distance < 0.5:
                            first_match_index = distances.index(min_distance)
                            name = known_face_names[first_match_index]
                            user_found = True
                            print(f"Utilisateur reconnu : {name}")
                            break

                    if user_found:
                        break

                    if cv2.waitKey(1) == 13:  # 13 est la touche Entrée
                        break

            except Exception as e:
                flash(f"Erreur: {str(e)}", "error")
                return redirect(url_for('login'))

            finally:
                cap.release()
                cv2.destroyAllWindows()

            if user_found:
                with sqlite3.connect(DATABASE) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM users WHERE username = ?', (name,))
                    user = cursor.fetchone()

                    if user:
                        session['user_id'] = user[0]
                        session['role'] = user[4]
                        flash('Connexion réussie !', 'success')
                        return redirect(url_for('dashboard'))
                    else:
                        flash('Utilisateur non trouvé.', 'error')
            else:
                flash("Utilisateur non reconnu. Veuillez vous inscrire.", "error")
                return redirect(url_for('register'))

    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    user = get_current_user()
    if not user:
        flash('Utilisateur non trouvé.', 'error')
        return redirect(url_for('login'))

    # Rediriger en fonction du rôle
    if user[4] == 'ADMIN':
        return redirect(url_for('admin_dashboard'))
    elif user[4] == 'EMPLOYEE':
        return redirect(url_for('employee_dashboard'))
    else:
        return render_template("dashboard.html", user=user)

@app.route('/admin/dashboard')
@role_required('ADMIN')
def admin_dashboard():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Récupérer la liste des utilisateurs
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()

        # Récupérer la liste des transactions
        cursor.execute('SELECT * FROM transactions')
        transactions = cursor.fetchall()

        # Récupérer la liste des prêts
        cursor.execute('SELECT * FROM loans')
        loans = cursor.fetchall()

        # Calculer le total des prêts
        cursor.execute('SELECT SUM(amount) FROM loans')
        total_loans = cursor.fetchone()[0] or 0  # Si aucun prêt, retourne 0

    return render_template("admin_dashboard.html", users=users, transactions=transactions, loans=loans, total_loans=total_loans)

@app.route('/admin/transactions')
@role_required('ADMIN')
def admin_transactions():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM transactions')
        transactions = cursor.fetchall()
    return render_template("admin_transactions.html", transactions=transactions)

@app.route('/admin/loans')
@role_required('ADMIN')
def admin_loans():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM loans')
        loans = cursor.fetchall()
    return render_template("admin_loans.html", loans=loans)

@app.route('/employee/dashboard')
@role_required('EMPLOYEE')
def employee_dashboard():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Récupérer la liste des utilisateurs
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()

        # Récupérer la liste des transactions
        cursor.execute('SELECT * FROM transactions')
        transactions = cursor.fetchall()

        # Récupérer la liste des factures
        cursor.execute('SELECT * FROM invoices')
        invoices = cursor.fetchall()

        # Récupérer la liste des prêts
        cursor.execute('SELECT * FROM loans')
        loans = cursor.fetchall()

    return render_template("employee_dashboard.html", users=users, transactions=transactions, invoices=invoices, loans=loans)

@app.route('/admin/users', methods=['GET', 'POST'])
@role_required('ADMIN')
def admin_users():
    if request.method == 'POST':
        username = request.form.get('username')
        password = generate_password_hash(request.form.get('password'))
        full_name = request.form.get('full_name')
        role = request.form.get('role')

        # Vérifier si l'utilisateur existe déjà
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash('Ce nom d\'utilisateur existe déjà. Veuillez en choisir un autre.', 'error')
                return redirect(url_for('admin_users'))

            # Si l'utilisateur n'existe pas, l'ajouter
            cursor.execute('INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, ?)',
                           (username, password, full_name, role))
            conn.commit()

        flash('Utilisateur ajouté avec succès.', 'success')
        return redirect(url_for('admin_users'))

    # Afficher la liste des utilisateurs
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()

    return render_template("admin_users.html", users=users)

@app.route('/admin/users/delete/<int:user_id>')
@role_required('ADMIN')
def admin_delete_user(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Récupérer le nom d'utilisateur avant de le supprimer
        cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        if not user:
            flash('Utilisateur non trouvé.', 'error')
            return redirect(url_for('admin_users'))

        username = user[0]

        # Supprimer les transactions associées
        cursor.execute('DELETE FROM transactions WHERE sender_id = ? OR receiver_id = ?', (user_id, user_id))

        # Supprimer les factures associées
        cursor.execute('DELETE FROM invoices WHERE user_id = ?', (user_id,))

        # Supprimer les prêts associés
        cursor.execute('DELETE FROM loans WHERE user_id = ?', (user_id,))

        # Supprimer l'utilisateur
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()

    # Supprimer les images de l'utilisateur du dossier dataset
    user_folder = os.path.join(DATASET_DIR, username)
    if os.path.exists(user_folder):
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            os.remove(image_path)
        os.rmdir(user_folder)

    # Recharger les visages connus
    load_known_faces()

    flash('Utilisateur supprimé avec succès.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/update_balance/<int:user_id>', methods=['GET', 'POST'])
@role_required('ADMIN')
def admin_update_balance(user_id):
    if request.method == 'POST':
        amount = float(request.form.get('amount'))  # Montant à ajouter ou retirer
        action = request.form.get('action')  # 'add' ou 'subtract'

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()

            # Récupérer le solde actuel de l'utilisateur
            cursor.execute('SELECT balance FROM users WHERE id = ?', (user_id,))
            current_balance = cursor.fetchone()[0]

            # Mettre à jour le solde en fonction de l'action
            if action == 'add':
                new_balance = current_balance + amount
            elif action == 'subtract':
                if current_balance >= amount:  # Vérifier que le solde est suffisant
                    new_balance = current_balance - amount
                else:
                    flash('Le solde est insuffisant pour effectuer cette opération.', 'error')
                    return redirect(url_for('admin_users'))

            # Mettre à jour le solde dans la base de données
            cursor.execute('UPDATE users SET balance = ? WHERE id = ?', (new_balance, user_id))

            # Journaliser l'opération
            cursor.execute('''
                INSERT INTO balance_history (user_id, amount, action, performed_by)
                VALUES (?, ?, ?, ?)
            ''', (user_id, amount, action.upper(), session['user_id']))

            conn.commit()

        flash(f'Solde mis à jour avec succès. Nouveau solde : {new_balance}', 'success')
        return redirect(url_for('admin_users'))

    # Afficher le formulaire de mise à jour du solde
    return render_template("update_balance.html", user_id=user_id, role='ADMIN')

@app.route('/admin/balance_history')
@role_required('ADMIN')
def admin_balance_history():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row  # Convertir les tuples en dictionnaires
        cursor = conn.cursor()
        cursor.execute('''
            SELECT bh.*, u.username as user_name, p.username as performed_by_name
            FROM balance_history bh
            JOIN users u ON bh.user_id = u.id
            JOIN users p ON bh.performed_by = p.id
        ''')
        history = cursor.fetchall()

    # Debug : Afficher les données récupérées dans la console
    print("Données récupérées :", history)

    return render_template("balance_history.html", history=history)

@app.route('/admin/pending_requests')
@role_required('ADMIN')
def admin_pending_requests():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row  # Convertir les tuples en dictionnaires
        cursor = conn.cursor()
        cursor.execute('''
            SELECT bh.*, u.username as user_name, p.username as performed_by_name
            FROM balance_history bh
            JOIN users u ON bh.user_id = u.id
            JOIN users p ON bh.performed_by = p.id
            WHERE bh.requires_approval = TRUE
        ''')
        pending_requests = cursor.fetchall()

    return render_template("admin_pending_requests.html", pending_requests=pending_requests)

@app.route('/admin/approve_request/<int:request_id>')
@role_required('ADMIN')
def admin_approve_request(request_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Récupérer la demande
        cursor.execute('SELECT * FROM balance_history WHERE id = ?', (request_id,))
        request_data = cursor.fetchone()

        if not request_data:
            flash('Demande non trouvée.', 'error')
            return redirect(url_for('admin_pending_requests'))

        # Vérifier si la demande nécessite une approbation
        if not request_data[6]:  # requires_approval est à l'index 6
            flash('Cette demande ne nécessite pas d\'approbation.', 'error')
            return redirect(url_for('admin_pending_requests'))

        user_id = request_data[1]
        amount = request_data[2]
        action = request_data[3]

        # Appliquer la modification
        if action == 'ADD':
            cursor.execute('UPDATE users SET balance = balance + ? WHERE id = ?', (amount, user_id))
        elif action == 'SUBTRACT':
            cursor.execute('UPDATE users SET balance = balance - ? WHERE id = ?', (amount, user_id))

        # Marquer la demande comme approuvée
        cursor.execute('UPDATE balance_history SET requires_approval = FALSE WHERE id = ?', (request_id,))
        conn.commit()

    flash('Demande approuvée avec succès.', 'success')
    return redirect(url_for('admin_pending_requests'))

@app.route('/admin/reject_request/<int:request_id>')
@role_required('ADMIN')
def admin_reject_request(request_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Récupérer la demande
        cursor.execute('SELECT * FROM balance_history WHERE id = ?', (request_id,))
        request_data = cursor.fetchone()

        if not request_data:
            flash('Demande non trouvée.', 'error')
            return redirect(url_for('admin_pending_requests'))

        # Vérifier si la demande nécessite une approbation
        if not request_data[6]:  # requires_approval est à l'index 6
            flash('Cette demande ne nécessite pas d\'approbation.', 'error')
            return redirect(url_for('admin_pending_requests'))

        # Marquer la demande comme refusée
        cursor.execute('DELETE FROM balance_history WHERE id = ?', (request_id,))
        conn.commit()

    flash('Demande refusée avec succès.', 'success')
    return redirect(url_for('admin_pending_requests'))

@app.route('/employee/loans/approve/<int:loan_id>')
@role_required('EMPLOYEE')
def employee_approve_loan(loan_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE loans SET status = "APPROVED" WHERE id = ?', (loan_id,))
        conn.commit()
    flash('Prêt approuvé avec succès.', 'success')
    return redirect(url_for('employee_dashboard'))

@app.route('/employee/loans/reject/<int:loan_id>')
@role_required('EMPLOYEE')
def employee_reject_loan(loan_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE loans SET status = "REJECTED" WHERE id = ?', (loan_id,))
        conn.commit()
    flash('Prêt rejeté avec succès.', 'success')
    return redirect(url_for('employee_dashboard'))

@app.route('/employee/invoices/pay/<int:invoice_id>')
@role_required('EMPLOYEE')
def employee_pay_invoice(invoice_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE invoices SET status = "PAID" WHERE id = ?', (invoice_id,))
        conn.commit()
    flash('Facture marquée comme payée.', 'success')
    return redirect(url_for('employee_dashboard'))

@app.route('/transfer', methods=['GET', 'POST'])
def transfer():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        receiver_username = request.form.get('receiver_username')
        amount = float(request.form.get('amount'))

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()

            # Vérifier si le destinataire existe
            cursor.execute('SELECT id, balance FROM users WHERE username = ?', (receiver_username,))
            receiver = cursor.fetchone()

            if not receiver:
                flash('Destinataire non trouvé.', 'error')
                return redirect(url_for('transfer'))

            # Vérifier le solde de l'expéditeur
            cursor.execute('SELECT balance FROM users WHERE id = ?', (user[0],))
            sender_balance = cursor.fetchone()[0]

            if sender_balance < amount:
                flash('Solde insuffisant.', 'error')
                return redirect(url_for('transfer'))

            # Mettre à jour les soldes
            cursor.execute('UPDATE users SET balance = balance - ? WHERE id = ?', (amount, user[0]))
            cursor.execute('UPDATE users SET balance = balance + ? WHERE id = ?', (amount, receiver[0]))

            # Enregistrer la transaction
            cursor.execute('INSERT INTO transactions (sender_id, receiver_id, amount) VALUES (?, ?, ?)',
                           (user[0], receiver[0], amount))
            conn.commit()

            flash('Transfert réussi !', 'success')
            return redirect(url_for('dashboard'))

    return render_template("transfer.html")
@app.route('/employee/users/update_balance/<int:user_id>', methods=['GET', 'POST'])
@role_required('EMPLOYEE')
def employee_update_balance(user_id):
    if request.method == 'POST':
        amount = float(request.form.get('amount'))  # Montant à ajouter ou retirer
        action = request.form.get('action')  # 'add' ou 'subtract'

        # Limite pour l'employé : ne peut pas modifier le solde de plus de 1000 € par jour
        if amount > 1000:
            flash('Vous ne pouvez pas modifier le solde de plus de 1000 € par jour.', 'error')
            return redirect(url_for('employee_dashboard'))

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()

            # Récupérer le solde actuel de l'utilisateur et la date de la dernière mise à jour
            cursor.execute('SELECT balance, last_balance_update FROM users WHERE id = ?', (user_id,))
            user_data = cursor.fetchone()
            current_balance = user_data[0]
            last_balance_update = user_data[1]

            # Vérifier si la dernière mise à jour a eu lieu aujourd'hui
            today = datetime.now().date()
            if last_balance_update == str(today):  # Convertir en chaîne pour la comparaison
                # Récupérer la somme totale des montants modifiés aujourd'hui par cet employé pour cet utilisateur
                cursor.execute('''
                    SELECT COALESCE(SUM(amount), 0) FROM balance_history
                    WHERE user_id = ? AND performed_by = ? AND DATE(timestamp) = ?
                ''', (user_id, session['user_id'], today))
                total_amount_today = cursor.fetchone()[0]

                # Vérifier si la somme totale dépasse 1000 €
                if total_amount_today + amount > 1000:
                    # Enregistrer la demande en attente d'approbation
                    cursor.execute('''
                        INSERT INTO balance_history (user_id, amount, action, performed_by, requires_approval)
                        VALUES (?, ?, ?, ?, TRUE)
                    ''', (user_id, amount, action.upper(), session['user_id']))
                    conn.commit()
                    flash('Vous avez atteint la limite de 1000 € par jour. Une demande d\'approbation a été envoyée à l\'admin.', 'warning')
                    return redirect(url_for('employee_dashboard'))

            # Mettre à jour le solde en fonction de l'action
            if action == 'add':
                new_balance = current_balance + amount
            elif action == 'subtract':
                if current_balance >= amount:  # Vérifier que le solde est suffisant
                    new_balance = current_balance - amount
                else:
                    flash('Le solde est insuffisant pour effectuer cette opération.', 'error')
                    return redirect(url_for('employee_dashboard'))

            # Mettre à jour le solde dans la base de données
            cursor.execute('UPDATE users SET balance = ?, last_balance_update = ? WHERE id = ?',
                           (new_balance, today, user_id))

            # Journaliser l'opération
            cursor.execute('''
                INSERT INTO balance_history (user_id, amount, action, performed_by, requires_approval)
                VALUES (?, ?, ?, ?, FALSE)
            ''', (user_id, amount, action.upper(), session['user_id']))

            conn.commit()

        flash(f'Solde mis à jour avec succès. Nouveau solde : {new_balance}', 'success')
        return redirect(url_for('employee_dashboard'))

    # Afficher le formulaire de mise à jour du solde
    return render_template("update_balance.html", user_id=user_id, role='EMPLOYEE')

@app.route('/invoices', methods=['GET', 'POST'])
def invoices():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        amount = float(request.form.get('amount'))
        description = request.form.get('description')

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO invoices (user_id, amount, description, status) VALUES (?, ?, ?, ?)',
                           (user[0], amount, description, 'PENDING'))
            conn.commit()

        flash('Facture créée avec succès.', 'success')
        return redirect(url_for('invoices'))

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM invoices WHERE user_id = ?', (user[0],))
        invoices = cursor.fetchall()

    return render_template("invoices.html", invoices=invoices)

@app.route('/loans', methods=['GET', 'POST'])
def loans():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        amount = float(request.form.get('amount'))
        interest_rate = float(request.form.get('interest_rate'))

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO loans (user_id, amount, interest_rate, status) VALUES (?, ?, ?, ?)',
                           (user[0], amount, interest_rate, 'PENDING'))
            conn.commit()

        flash('Demande de prêt soumise avec succès.', 'success')
        return redirect(url_for('loans'))

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM loans WHERE user_id = ?', (user[0],))
        loans = cursor.fetchall()

    return render_template("loans.html", loans=loans)

@app.route('/logout')
def logout():
    session.clear()
    flash('Déconnexion réussie.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)