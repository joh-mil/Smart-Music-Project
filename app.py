from fileinput import filename
from difflib import SequenceMatcher
from posixpath import sep
from flask import Flask, render_template, request, redirect, session, send_file, url_for
from librosa.feature import tempo
from matplotlib import lines
from pydub import AudioSegment

try:
    from mutagen import File as MutagenFile
except Exception:
    MutagenFile = None  

import sys
import subprocess
import csv
import os
import shutil
import sqlite3
import requests
import librosa
import numpy as np
import joblib
import mutagen

genre_model = joblib.load("genre_model.pkl")
from datetime import datetime

app = Flask(__name__)
# Tell Flask to use your existing 'uploads' directory
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = "smart_music_secret_key"

# Load the trained genre model
GENRE_MODEL_PATH = "genre_model.pkl"
genre_model = None
if os.path.exists(GENRE_MODEL_PATH):
    try:
        genre_model = joblib.load(GENRE_MODEL_PATH)
        print("Genre AI Model loaded successfully!")
    except Exception as e:
        print("Error loading genre model:", e)

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

FFMPEG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")

AudioSegment.converter = os.path.join(FFMPEG_FOLDER, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_FOLDER, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_FOLDER, "ffprobe.exe")
os.environ["PATH"] = FFMPEG_FOLDER + ";" + os.environ["PATH"]

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/songs", exist_ok=True)
os.makedirs("static/lyrics", exist_ok=True)
os.makedirs("static/recordings", exist_ok=True)
os.makedirs("static/karaoke_clips", exist_ok=True)
os.makedirs("static/generated/mashups", exist_ok=True)
os.makedirs("static/generated/bgm", exist_ok=True)
os.makedirs("static/generated/karaoke", exist_ok=True)
os.makedirs("static/feature_uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

FIELDNAMES = [
    "filename", "song_name", "singer", "language",
    "genre", "mood", "lyrics", "uploaded_by", "karaoke_start", "karaoke_end",
    "ai_genre", "ai_mood", "tag", "tempo", "energy", "favorite"
]

def load_songs():
    global songs
    songs = []
    if os.path.exists("songs.csv"):
        with open("songs.csv", "r", encoding="utf-8") as f:
            # DictReader matches data by column name, preventing shifting bugs
            reader = csv.DictReader(f)
            for row in reader:
                # Ensure boolean/string evaluation stays clean
                row["favorite"] = row.get("favorite") == "True"
                try:
                    row["tempo"] = float(row.get("tempo", 120.0))
                    row["energy"] = float(row.get("energy", 0.05))
                except ValueError:
                    row["tempo"] = 120.0
                    row["energy"] = 0.05
                songs.append(row)

def save_songs_csv():
    with open("songs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for song in songs:
            # Create a copy to modify formatting safely for storage
            song_copy = song.copy()
            song_copy["favorite"] = str(song_copy.get("favorite", False))
            writer.writerow(song_copy)


def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        login_count INTEGER DEFAULT 0,
        upload_count INTEGER DEFAULT 0,
        download_count INTEGER DEFAULT 0,
        karaoke_count INTEGER DEFAULT 0,
        mashup_count INTEGER DEFAULT 0,
        bgm_count INTEGER DEFAULT 0
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        feature_name TEXT,
        used_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        message TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_downloads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        filename TEXT,
        file_type TEXT,
        downloaded_at TEXT
    )
    """)


    cursor.execute("PRAGMA table_info(users)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    extra_columns = {
        "login_count": "INTEGER DEFAULT 0",
        "upload_count": "INTEGER DEFAULT 0",
        "download_count": "INTEGER DEFAULT 0",
        "karaoke_count": "INTEGER DEFAULT 0",
        "mashup_count": "INTEGER DEFAULT 0",
        "bgm_count": "INTEGER DEFAULT 0"
    }

    for column, column_type in extra_columns.items():
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {column} {column_type}")

    conn.commit()
    conn.close()


def track_feature(feature_name):
    username = session.get("user", "admin" if session.get("admin") else "guest")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feature_usage (username, feature_name, used_at) VALUES (?, ?, ?)",
        (username, feature_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def increment_user_count(username, column_name):
    allowed = [
        "login_count", "upload_count", "download_count",
        "karaoke_count", "mashup_count", "bgm_count"
    ]

    if not username or column_name not in allowed:
        return

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE users SET {column_name} = {column_name} + 1 WHERE username=?",
        (username,)
    )
    conn.commit()
    conn.close()


def create_csv_if_missing():
    if not os.path.exists("songs.csv"):
        with open("songs.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()


def load_songs_csv():
    loaded_songs = []
    create_csv_if_missing()

    with open("songs.csv", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            clean_row = {}
            for field in FIELDNAMES:
                clean_row[field] = row.get(field, "")
            loaded_songs.append(clean_row)

    return loaded_songs


def save_songs_csv():
    with open("songs.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for song in songs:
            clean_song = {}
            for field in FIELDNAMES:
                clean_song[field] = song.get(field, "")
            writer.writerow(clean_song)


def login_required():
    return session.get("user") or session.get("admin")


def current_username():
    return session.get("user")


def user_can_access_song(song):
    if session.get("admin"):
        return True

    username = current_username()
    return song.get("uploaded_by") == username or song.get("uploaded_by") == "Admin"


def get_visible_songs():
    return [
        song for song in songs 
        if user_can_access_song(song) 
        and not song.get("filename", "").startswith(("bgm_", "mashup_", "karaoke_mix_"))
        and " BGM" not in song.get("song_name", "")
    ]


def auto_detect_genre_mood(audio_path):
    try:
        import numpy as np
        import librosa

        y, sr = librosa.load(audio_path, duration=30)

        # =========================
        # 1. FEATURE EXTRACTION
        # =========================

        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T,
            axis=0
        )

        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        tempo = float(tempo)

        energy = librosa.feature.rms(y=y)

        # FINAL FEATURE VECTOR (must match training)
        feature_vector = np.hstack([mfcc, tempo, energy]).reshape(1, -1)

        # =========================
        # 2. GENRE PREDICTION (ML MODEL)
        # =========================

        genre = "Unknown"

        if genre_model is not None:
            genre = genre_model.predict(feature_vector)[0]

        # =========================
        # 3. MOOD PREDICTION (RULE BASED)
        # =========================

        if tempo > 130 and energy > 0.045:
            mood = "Energetic"
        elif 100 <= tempo <= 130:
            mood = "Happy"
        elif 70 <= tempo < 100:
            mood = "Calm"
        elif energy < 0.03:
            mood = "Sad"
        else:
            mood = "Romantic"

        return genre, mood

    except Exception as e:
        print("AI detection error:", e)
        return "Unknown", "Unknown"


def extract_song_details(filename_without_ext, audio_path=None):
    clean_name = filename_without_ext.replace("_", " ").strip()

    title = clean_name.title() if clean_name else "Unknown Title"
    artist = "Unknown"

    if audio_path and MutagenFile:
        try:
            audio_info = MutagenFile(audio_path, easy=True)
            if audio_info:
                meta_title = audio_info.get("title", [""])[0]
                meta_artist = audio_info.get("artist", [""])[0]

                if meta_title:
                    title = meta_title.strip().title()
                if meta_artist:
                    artist = meta_artist.strip().title()
        except Exception:
            pass

    if artist == "Unknown":
        separators = [
            " - ", " by ", " singer ", " artist ", " music by ",
            " sung by ", " vocals by ", " ft ", " feat ",
            " featuring ", " composer ", " composers ",
            " contributing artist "
        ]

        lower_name = clean_name.lower()

        for sep in separators:
            if sep in lower_name:
                index = lower_name.find(sep)
                title = clean_name[:index].strip().title()
                artist = clean_name[index + len(sep):].strip().title()
                break

    if not title:
        title = clean_name.title() if clean_name else "Unknown Title"

    if not artist:
        artist = "Unknown"

    return title, artist

def clean_lyrics_name(name):
    name = name.lower()

    # remove extensions
    name = name.replace(".mp3", "")
    name = name.replace(".wav", "")
    name = name.replace(".m4a", "")
    name = name.replace(".txt", "")
    name = name.replace(".lrc", "")

    # remove website/common junk
    junk_words = [
        "sensongsmp3", "sensongs", "naasongs", "pagalworld",
        "download", "song", "songs", "official", "lyrical"
    ]

    for word in junk_words:
        name = name.replace(word, "")

    # symbols cleanup
    name = name.replace("::", " ")
    name = name.replace("-", " ")
    name = name.replace("_", " ")
    name = name.replace(".", " ")

    name = " ".join(name.split())
    return name.replace(" ", "_")

def clean_song_identity(text):
    text = text.lower()
    text = text.replace(".mp3", "").replace(".wav", "").replace(".m4a", "")
    text = text.replace("_", " ").replace("-", " ").replace("::", " ")
    text = " ".join(text.split())
    return text


def find_admin_lyrics_by_song_name(song_name):
    lyrics_folder = os.path.join("static", "lyrics")
    os.makedirs(lyrics_folder, exist_ok=True)

    song_key = clean_lyrics_name(song_name)

    for file in os.listdir(lyrics_folder):
        if not file.lower().endswith(".txt",".lrc"):
            continue

        file_key = clean_lyrics_name(os.path.splitext(file)[0])

        if file_key == song_key:
            with open(os.path.join(lyrics_folder, file), "r", encoding="utf-8") as f:
                return f.read()

    return ""

def convert_lrc_to_seconds(lyrics_text, start_sec=0, end_sec=None):

    import re

    print("FUNCTION START_SEC =", start_sec)

    start_sec = float(start_sec or 0)
    end_sec = float(end_sec or 999999)

    final_lines = []

    for line in lyrics_text.splitlines():

        match = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)

        if not match:
            continue

        minutes = int(match.group(1))
        seconds = float(match.group(2))
        text = match.group(3).strip()

        full_time = (minutes * 60) + seconds

        print("FULL TIME =", full_time)

        if start_sec <= full_time <= end_sec and text:

            clip_time = round(full_time - start_sec, 2)

            print("CLIP TIME =", clip_time)

            final_lines.append(f"{clip_time}|{text}")

    return "\n".join(final_lines)


def fetch_lyrics_from_api(song_name, singer):
    # API disabled to keep the project local/offline
    return ""


def convert_transcript_to_karaoke(text, start_sec=0, end_sec=None):
    final_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("(") and ")" in line:
            try:
                time_part = line.split(")")[0].replace("(", "").strip()
                lyric = line.split(")", 1)[1].strip()

                mins, secs = time_part.split(":")
                original_time = int(mins) * 60 + int(secs)

                if end_sec is not None:
                    if original_time < start_sec or original_time > end_sec:
                        continue

                karaoke_time = original_time - start_sec

                if karaoke_time < 0:
                    karaoke_time = 0

                else:
                    karaoke_time = original_time

                final_lines.append(f"{karaoke_time}|{lyric}")

            except:
                pass

    return "\n".join(final_lines)

def list_generated_files(folder_name):
    folder_path = os.path.join("static", "generated", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            files.append({
                "filename": filename,
                "url": "/static/generated/" + folder_name + "/" + filename,
                "folder": folder_name
            })

    files.reverse()
    return files


def predict_mashup_mood(mood1, mood2):
    if mood1 == mood2:
        return mood1
    if "Happy" in [mood1, mood2] and "Energetic" in [mood1, mood2]:
        return "Party Mood"
    if "Sad" in [mood1, mood2]:
        return "Emotional Mix"
    return "Mixed Mood"


def check_mashup_compatibility(song1, song2):
    same_genre = song1.get("genre") == song2.get("genre")
    same_mood = song1.get("mood") == song2.get("mood")

    if same_genre and same_mood:
        return {
            "score": 100,
            "level": "Excellent",
            "emoji": "🔥",
            "reason": "Same genre and same mood. Perfect for mashup.",
            "can_mashup": True
        }

    if same_genre:
        return {
            "score": 75,
            "level": "Good",
            "emoji": "✅",
            "reason": "Same genre. Mashup is suitable.",
            "can_mashup": True
        }

    if same_mood:
        return {
            "score": 65,
            "level": "Average",
            "emoji": "⚠️",
            "reason": "Same mood. Mashup can be tried.",
            "can_mashup": True
        }

    return {
        "score": 30,
        "level": "Bad",
        "emoji": "❌",
        "reason": "Genre and mood are both different. Select at least same genre or same mood.",
        "can_mashup": False
    }


def create_mashup(file1, file2, start1, start2, song1_intro, song2_duration, song1_ending, crossfade, mashup_type="smooth"):
    if not os.path.exists(file1) or not os.path.exists(file2):
        return None

    mashup_type = (mashup_type or "smooth").lower()

    song1 = AudioSegment.from_file(file1)
    song2 = AudioSegment.from_file(file2)

    part1_intro = song1[start1:start1 + song1_intro]
    part2_main = song2[start2:start2 + song2_duration]

    ending_start = start1 + song1_intro
    part1_ending = song1[ending_start:ending_start + song1_ending]

    if len(part1_ending) < 3000:
        part1_ending = song1[start1:start1 + song1_ending]

    def normalize(audio, target_db=-16):
        if audio.dBFS != float("-inf"):
            return audio.apply_gain(target_db - audio.dBFS)
        return audio

    part1_intro = normalize(part1_intro)
    part2_main = normalize(part2_main)
    part1_ending = normalize(part1_ending)

    # ---------------- SMOOTH MASHUP ----------------
    if mashup_type == "smooth":
        part1_intro = part1_intro.fade_in(1500).fade_out(crossfade)
        part2_main = part2_main.fade_in(crossfade).fade_out(crossfade)
        part1_ending = part1_ending.fade_in(crossfade).fade_out(2500)

    # ---------------- DJ BEAT MASHUP ----------------
    elif mashup_type == "dj":
        part1_intro = part1_intro + 2
        part2_main = part2_main + 4
        part1_ending = part1_ending + 2

        part1_intro = part1_intro.fade_in(500).fade_out(1000)
        part2_main = part2_main.fade_in(500).fade_out(1000)
        part1_ending = part1_ending.fade_in(500).fade_out(1200)

        crossfade = min(crossfade, 1200)

    # ---------------- ROMANTIC MASHUP ----------------
    elif mashup_type == "romantic":
        part1_intro = part1_intro - 2
        part2_main = part2_main - 1
        part1_ending = part1_ending - 2

        part1_intro = part1_intro.fade_in(2500).fade_out(crossfade)
        part2_main = part2_main.fade_in(crossfade).fade_out(crossfade)
        part1_ending = part1_ending.fade_in(crossfade).fade_out(3500)

    # ---------------- PARTY MASHUP ----------------
    elif mashup_type == "party":
        part1_intro = part1_intro + 3
        part2_main = part2_main + 4
        part1_ending = part1_ending + 3

        part1_intro = part1_intro.fade_in(700).fade_out(1500)
        part2_main = part2_main.fade_in(1000).fade_out(1200)
        part1_ending = part1_ending.fade_in(800).fade_out(1800)

        crossfade = min(crossfade, 1800)

    # ---------------- CHILL MASHUP ----------------
    elif mashup_type == "chill":
        part1_intro = part1_intro - 3
        part2_main = part2_main - 2
        part1_ending = part1_ending - 3

        part1_intro = part1_intro.fade_in(3000).fade_out(crossfade)
        part2_main = part2_main.fade_in(crossfade).fade_out(crossfade)
        part1_ending = part1_ending.fade_in(crossfade).fade_out(4000)

    # ---------------- AUTO AI MASHUP ----------------
    elif mashup_type == "auto":
        part1_intro = part1_intro.fade_in(1500).fade_out(crossfade)
        part2_main = part2_main.fade_in(crossfade).fade_out(crossfade)
        part1_ending = part1_ending.fade_in(crossfade).fade_out(2500)

    else:
        part1_intro = part1_intro.fade_in(1500).fade_out(crossfade)
        part2_main = part2_main.fade_in(crossfade).fade_out(crossfade)
        part1_ending = part1_ending.fade_in(crossfade).fade_out(2500)

    final_mix = part1_intro.append(part2_main, crossfade=crossfade)
    final_mix = final_mix.append(part1_ending, crossfade=crossfade)

    os.makedirs(os.path.join("static", "generated", "mashups"), exist_ok=True)

    unique_name = "mashup_" + mashup_type + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp3"
    output_path = os.path.join("static", "generated", "mashups", unique_name)

    final_mix.export(output_path, format="mp3", bitrate="192k")

    return output_path

def extract_song_and_singer(original_name, file_path):
    """Extracts clean Song Name and Singer Name from metadata tags or filename."""
    title = os.path.splitext(original_name)[0]
    artist = "Unknown Artist"
    
    if " - " in original_name:
        parts = original_name.split(" - ", 1)
        title = parts[0].strip()
        artist = os.path.splitext(parts[1])[0].strip()
    elif " :: " in original_name:
        parts = original_name.split(" :: ", 1)
        title = parts[0].strip()
        artist = os.path.splitext(parts[1])[0].strip()

    try:
        audio = mutagen.File(file_path)
        if audio is not None:
            for key in ['title', 'TIT2', 'album', 'TALB']:
                if key in audio and str(audio[key][0]).strip():
                    title = str(audio[key][0]).strip()
                    break
            for key in ['artist', 'TPE1', 'albumartist', 'TPE2', 'contributing_artist']:
                if key in audio and str(audio[key][0]).strip():
                    artist = str(audio[key][0]).strip()
                    break
    except:
        pass

    if "sensongsemp3" in artist.lower() or "sensongs" in artist.lower():
        artist = "Unknown Artist"
    if "sensongsemp3" in title.lower() or "sensongs" in title.lower():
        title = title.split(" :: ")[0].strip()
        
    return title, artist

def map_mood(genre_name):
    """Simple mapping rules to instantly match mood to genre choice."""
    genre_lower = genre_name.lower()
    if genre_lower in ['classical', 'jazz', 'blues', 'melody', 'romantic']:
        return "Calm"
    elif genre_lower in ['rock', 'metal', 'mass', 'hiphop']:
        return "Energetic"
    else:
        return "Happy"
    
# --- ML MODEL LOADING ---
try:
    genre_model = joblib.load('genre_model.pkl')
    genre_scaler = joblib.load('genre_scaler.pkl')
    genre_encoder = joblib.load('genre_encoder.pkl')
except:
    genre_model, genre_scaler, genre_encoder = None, None, None

# --- AUDIO & METADATA HELPERS ---
def extract_song_and_singer(original_name, file_path):
    title = os.path.splitext(original_name)[0]
    artist = "Unknown Artist"
    if " - " in original_name:
        parts = original_name.split(" - ", 1)
        title = parts[0].strip()
        artist = os.path.splitext(parts[1])[0].strip()
    elif " :: " in original_name:
        parts = original_name.split(" :: ", 1)
        title = parts[0].strip()
        artist = os.path.splitext(parts[1])[0].strip()
    try:
        audio = mutagen.File(file_path)
        if audio is not None:
            for key in ['title', 'TIT2', 'album', 'TALB']:
                if key in audio and str(audio[key][0]).strip():
                    title = str(audio[key][0]).strip()
                    break
            for key in ['artist', 'TPE1', 'albumartist', 'TPE2', 'contributing_artist']:
                if key in audio and str(audio[key][0]).strip():
                    artist = str(audio[key][0]).strip()
                    break
    except:
        pass
    if "sensongsemp3" in artist.lower() or "sensongs" in artist.lower():
        artist = "Unknown Artist"
    return title, artist

def predict_genre(filename):
    try:
        # 📊 Load dataset
        data = pd.read_csv(r"C:\Users\johnm\Downloads\Smart Music_Project\features_30_sec.csv")

        # 🔍 Find song
        song_data = data[data['filename'].str.contains(filename, na=False)]

        if song_data.empty:
            print("Song not found in dataset:", filename)
            return "Unknown"

        # 🎯 Extract features
        features = song_data.drop(columns=['filename', 'length', 'label'])

        # ⚡ Ensure correct format
        features = features.values

        # 🧠 Scale
        features_scaled = scaler.transform(features)

        # 🤖 Predict
        prediction = model.predict(features_scaled)[0]

        # 🎵 Map result
        return genre_mapping.get(prediction, prediction)

    except Exception as e:
        print("Prediction error:", e)
        return "Unknown"

def map_mood(genre_name):
    genre_lower = genre_name.lower()
    if genre_lower in ['classical', 'jazz', 'blues', 'melody', 'romantic']:
        return "Calm"
    elif genre_lower in ['rock', 'metal', 'mass', 'hiphop']:
        return "Energetic"
    else:
        return "Happy"

# --- CSV STORAGE HELPERS ---
def read_songs():
    songs = []
    if os.path.exists('songs.csv'):
        with open('songs.csv', mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append(row)
    return songs

def write_songs(songs):
    fieldnames = ['filename', 'song_name', 'singer', 'genre', 'mood']
    with open('songs.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for song in songs:
            writer.writerow(song) 
@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin"] = username
            session.pop("user", None)
            return redirect("/admin-dashboard")
        else:
       
            return render_template("admin_login.html", error="Invalid Admin Username or Password")

@app.route("/admin-dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect("/admin-login")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedback")
    total_feedbacks = cursor.fetchone()[0]

    conn.close()

    total_songs = len(songs)
    mashup_count = len(list_generated_files("mashups"))
    karaoke_count = len(list_generated_files("karaoke"))
    bgm_count = len(list_generated_files("bgm"))

  
    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_songs=total_songs,
        mashup_count=mashup_count,
        karaoke_count=karaoke_count,
        bgm_count=bgm_count,
        total_feedbacks=total_feedbacks
    )


@app.route("/admin-users")
def admin_users():
    if not session.get("admin"):
        return redirect("/admin-login")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
     
    SELECT id, username, password,
               login_count,
               upload_count,
               download_count,
               karaoke_count,
               mashup_count,
               bgm_count
    
     FROM users
        ORDER BY id DESC
    """)

    users = cursor.fetchall()
    conn.close()

    return render_template("admin_users.html", users=users)


@app.route("/admin-songs")
def admin_songs():
    if not session.get("admin"):
        return redirect("/admin-login")

    return render_template("admin_songs.html", songs=songs)


@app.route("/admin-logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin-login")


@app.route("/delete-user/<int:user_id>")
def delete_user(user_id):
    if not session.get("admin"):
        return redirect("/admin-login")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
   
    cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    return redirect("/admin-users")


@app.route("/admin-upload-song", methods=["GET", "POST"])
def admin_upload_song():
    if not session.get("admin"):
        return redirect("/admin-login")

    if request.method == "POST":
        files = request.files.getlist("audio_files")

        for file in files:
            if file.filename == "":
                continue

    
            original_filename = file.filename
            safe_filename = original_filename.replace(" ", "_").replace(":", "")

            # Keep original name for metadata extraction (before adding prefix)
            original_name, ext = os.path.splitext(safe_filename)

            unique_prefix = "Admin_" + datetime.now().strftime("%Y%m%d_%H%M%S_")
            safe_filename = unique_prefix + safe_filename

     
            temp_path = os.path.join("uploads", safe_filename)
            file.save(temp_path)

            # Use original_name (no prefix) so artist is not polluted
            title, artist = extract_song_details(original_name, temp_path)

            if ext.lower() == ".mp4":
                audio = AudioSegment.from_file(temp_path, format="mp4")
       
                new_filename = os.path.splitext(safe_filename)[0] + ".wav"
                new_path = os.path.join("static", "songs", new_filename)
                audio.export(new_path, format="wav")
                os.remove(temp_path)
            else:
                new_filename = safe_filename
    
                new_path = os.path.join("static", "songs", new_filename)
                shutil.copy2(temp_path, new_path)
                os.remove(temp_path)

            # 1. Run your AI function to detect genre and mood
            genre, mood = auto_detect_genre_mood(new_path)

            # 2. Extract tempo and energy features directly for this song
            try:
                y, sr = librosa.load(new_path, duration=30)
                
                # Fetch tempo
                tempo_data = librosa.beat.beat_track(y=y, sr=sr)[0]
       
                # Handle cases where tempo might return as an array or a single float
                tempo_val = float(tempo_data[0]) if hasattr(tempo_data, "__len__") else float(tempo_data)
                
                # Fetch energy
                energy_matrix = librosa.feature.rms(y=y)
   
                energy_val = float(np.mean(energy_matrix[0]))
            except Exception as e:
                print(f"Error extracting tempo/energy details: {e}")
                tempo_val = 120.0
                energy_val = 0.05

            # 3. Safely append to the songs array with clean dictionary formatting
            songs.append({
                "song_name": title,
                "artist": artist,
                "filename": new_filename,

                # 🎯 AI OUTPUT (internal)
        
                "ai_genre": genre,
                "ai_mood": mood,

                # 🧠 USER FINAL CONTROL (editable)
                "genre": genre,        
                "tag": "",           
   

                # 🔥 AI AUDIO FEATURES (Fixed Syntax)
                "tempo": tempo_val,
                "energy": energy_val,

                # ⭐ USER FEATURES
                "favorite": False,

      
                # ⚡ SYSTEM FLAGS
                "uploaded_by": session.get("user", "Unknown")
            })

        save_songs_csv()
        track_feature("Admin Upload Song")
        return redirect("/admin-songs")

    return render_template("admin_upload_song.html", songs=songs)




@app.route("/admin-train")
def admin_train():
    if not session.get("admin"):
        return redirect("/admin-login")

    return render_template("admin_train.html")


@app.route("/admin-settings")
def admin_settings():
  
    if not session.get("admin"):
        return redirect("/admin-login")

    return render_template("admin_settings.html")

@app.route("/")
def intro_page():
    return render_template("intro.html")

@app.route("/abstract")
def abstract_page():
    return render_template("abstract.html")

@app.route("/start")
def start_page():
    return render_template("start.html")

@app.route("/shutdown", methods=["POST"])
def shutdown():
    os._exit(0)
    return "Shutting down..."


@app.route("/home")
def home():
    if not session.get("user"):
        return redirect("/login")

    track_feature("Home Page")

    search_text = request.args.get("search", "").lower()
    genre = request.args.get("genre")
    mood = request.args.get("mood")

    filtered_songs = get_visible_songs()

    if search_text:
 
        filtered_songs = [
            s for s in filtered_songs
            if search_text in s.get("song_name", "").lower()
            or search_text in s.get("singer", "").lower()
            or search_text in s.get("genre", "").lower()
            or search_text in s.get("mood", "").lower()
        ]

   
    if genre:
        filtered_songs = [s for s in filtered_songs if s.get("genre") == genre]

    if mood:
        filtered_songs = [s for s in filtered_songs if s.get("mood") == mood]

    return render_template(
        "index.html",
        songs=filtered_songs,
        mashups=list_generated_files("mashups"),
        bgm_files=list_generated_files("bgm"),
        karaoke_files=list_generated_files("karaoke"),
        username=session.get("user")
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            return render_template("register.html", error="Username and password required")

        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username already exists")

        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
  
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = cursor.fetchone()

        if user:
            cursor.execute(
                "UPDATE users SET login_count = login_count + 1 WHERE username=?",
                (username,)
       
            )
            conn.commit()
            conn.close()

            session.pop("admin", None)
            session["user"] = username
            track_feature("User Login")
            return redirect("/home")

        conn.close()
        return render_template("login.html", error="Invalid Username or Password")

    return render_template("login.html")


@app.route("/download_file/<file_type>/<filename>")
def download_file(file_type, filename):
    if not login_required():
        return redirect("/login")

    username = session.get("user")
    
    # Track the download in DB
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Check if already tracked to avoid duplicates
    cursor.execute(
        "SELECT 1 FROM user_downloads WHERE username=? AND filename=?",
        (username, filename)
    )
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO user_downloads (username, filename, file_type, downloaded_at) VALUES (?, ?, ?, ?)",
            (username, filename, file_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
    conn.close()
    
    # Record in general feature usage
    track_feature(f"Download {file_type.title()}")
    increment_user_count(username, "download_count")

    # Determine file path
    if file_type == "bgm":
        file_path = os.path.join("static", "generated", "bgm", filename)
    elif file_type == "mashup":
        file_path = os.path.join("static", "generated", "mashups", filename)
    elif file_type == "karaoke":
        file_path = os.path.join("static", "generated", "karaoke", filename)
    else:
        file_path = os.path.join("static", "songs", filename)

    if os.path.exists(file_path):
  
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


@app.route("/my_downloads")
def my_downloads():
    if not login_required():
        return redirect("/login")

    username = session.get("user")
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, file_type, downloaded_at FROM user_downloads WHERE username=? ORDER BY downloaded_at DESC",
        (username,)
    )
    downloads = cursor.fetchall()
    conn.close()

    downloaded_files = []
    for download_id, filename, file_type, downloaded_at in downloads:
        url = ""
        if file_type == "bgm":
            url = "/static/generated/bgm/" + filename
        elif file_type == "mashup":
           
            url = "/static/generated/mashups/" + filename
        elif file_type == "karaoke":
            url = "/static/generated/karaoke/" + filename
        else:
            url = "/static/songs/" + filename
            
        downloaded_files.append({
            "id": download_id,
           
            "filename": filename,
            "file_type": file_type.title(),
            "url": url,
            "downloaded_at": downloaded_at
        })

    return render_template("my_downloads.html", downloads=downloaded_files)


@app.route("/delete_download/<int:download_id>")
def delete_download(download_id):
    if not login_required():
        return redirect("/login")

    username = session.get("user")
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Ensure the user owns the download record before deleting
    cursor.execute(
        "DELETE FROM user_downloads WHERE id=? AND username=?",
        (download_id, username)
    )
    conn.commit()
    conn.close()
    
    return redirect("/my_downloads")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        files = request.files.getlist('audio_files')
        
        # Capture form metadata and custom choices from upload.html
        form_song_name = request.form.get('song_name', '').strip()
        form_singer = request.form.get('singer', '').strip()
        user_selected_genre = request.form.get('user_genre', '').strip()
        
        songs = read_songs()
        duplicate_detected = False
        
        for file in files:
            if file and file.filename:
                # 1. Protection Check against Duplicates
                if any(s['filename'] == file.filename for s in songs):
                    duplicate_detected = True
                    continue  # Skip duplicate file processing securely
                
                # 2. File Persistence Storage
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
      
                # 3. Audio Extraction Fallbacks
                extracted_name, extracted_singer = extract_song_and_singer(file.filename, file_path)
                final_song_name = form_song_name if form_song_name else extracted_name
                final_singer = form_singer if form_singer else extracted_singer
                
                # 4. Predict / Train Genre & Map Mood properties
                ai_predicted_genre = predict_genre(file_path)
                
                # Prioritize manual user dropdown override unless set to 'auto'
                if user_selected_genre and user_selected_genre != "auto":
                    final_genre = user_selected_genre
                else:
                    final_genre = ai_predicted_genre
                    
                final_mood = map_mood(final_genre)
              
                # 5. Save everything neatly to database array dictionary
                songs.append({
                    'filename': file.filename,
                    'song_name': final_song_name,
                    'singer': final_singer,
                    'genre': final_genre,
                    'mood': final_mood,
                    'uploaded_by': session.get('user', 'Unknown')
                })
        
        # Commit changes to CSV database storage 
        write_songs(songs)
       
        # 6. Smooth Redirect straight back to Home page (index.html screen view)
        if duplicate_detected:
            # If a duplicate occurred, render home screen with explicit layout message
            return render_template('index.html', songs=songs, message="⚠️ Duplicate song files were skipped!")
            
        return redirect(url_for('home'))

    # GET Request: Renders the beautiful upload configuration console panel layout
    return render_template('upload.html', songs=read_songs())

@app.route('/update_song/<filename>', methods=['POST'], endpoint='manual_genre_update_route')
def handle_manual_genre_update(filename):
    new_genre = request.form.get('genre')
   
    songs = read_songs()
    
    for song in songs:
        if song['filename'] == filename:
            song['genre'] = new_genre
            song['mood'] = map_mood(new_genre)
            break
            
    write_songs(songs)
    # Change 'index' back to your actual function name:
    return redirect(url_for('upload_page')) # <-- We will update this next!

@app.route("/search", methods=["GET", "POST"])
def search():
    if not login_required():
        return redirect("/login")

    track_feature("Search")

    query = request.args.get("q", "").lower()
    results = []

    if query:
        results = [
            song for song in get_visible_songs()
            if query in song.get("song_name", "").lower()
            
            or query in song.get("singer", "").lower()
            or query in song.get("genre", "").lower()
            or query in song.get("mood", "").lower()
        ]

    return render_template("search.html", results=results, query=query)


@app.route("/karaoke/<filename>")
def karaoke(filename):
    print("KARAOKE ROUTE OPENED")

    if not login_required():
        return redirect("/login")

    selected_song = next(
        (song for song in songs if song.get("filename") == filename and user_can_access_song(song)),
   
        None
    )

    if not selected_song:
        return "Song not found"

    lyrics_folder = os.path.join("static", "lyrics")
    os.makedirs(lyrics_folder, exist_ok=True)

    selected_song["lyrics"] = ""

    song_name_clean = clean_lyrics_name(selected_song.get("song_name", ""))

    for file in os.listdir(lyrics_folder):
        
        if not file.lower().endswith((".txt", ".lrc")):
            continue
        

   
        file_clean = clean_lyrics_name(os.path.splitext(file)[0])

        print("SONG CLEAN:", song_name_clean)
        print("FILE CLEAN:", file_clean)

        if (
            file_clean == song_name_clean
            or file_clean in song_name_clean
            or song_name_clean in file_clean
        ):
            
            print("MATCHED LYRICS:", file)

            lyrics_path = os.path.join(lyrics_folder, file)

            with open(lyrics_path, "r", encoding="utf-8") as f:
                lyrics_text = f.read()

            if file.lower().endswith(".lrc"):
                start_sec = float(request.args.get("start", 0))
                
                end_sec = float(request.args.get("end", 999999))

                selected_song["lyrics"] = convert_lrc_to_seconds(
                    lyrics_text,
                    start_sec,
                    end_sec
                )
   
                 
            elif "(" in lyrics_text and ")" in lyrics_text:
                selected_song["lyrics"] = convert_transcript_to_karaoke(
                    lyrics_text
                )

           
            else:
                selected_song["lyrics"] = lyrics_text

            print("LOADED LYRICS:", selected_song["lyrics"][:100])
            break

    track_feature("Karaoke Page")
    increment_user_count(session.get("user"), "karaoke_count")

    return render_template("karaoke.html", song=selected_song)
    
def convert_txt_to_karaoke(lyrics_text, preview_duration):
    lines = []

    for line in lyrics_text.splitlines():
        line = line.strip()

        if line:
 
            lines.append(line)

    if not lines:
        return ""

    time_gap = preview_duration / len(lines)

    final_lines = []

    for index, line in enumerate(lines):
        timestamp = round(index * time_gap, 2)
        final_lines.append(f"{timestamp}|{line}")

    return "\n".join(final_lines)

@app.route("/prepare_karaoke/<filename>", methods=["POST"])
def prepare_karaoke(filename):
    print("PREPARE KARAOKE FUNCTION STARTED")

    if not login_required():
        return redirect("/login")

  
    selected_song = next(
        (song for song in songs if song.get("filename") == filename and user_can_access_song(song)),
        None
    )

    if not selected_song:
        return "You cannot access this file"

    start_sec = int(request.form.get("start_time") or 0)
    end_sec = int(request.form.get("end_time") or 30)

    print("PREVIEW START =", start_sec)
    print("PREVIEW END =", end_sec)

    if end_sec <= start_sec:
        return "End time must be greater than start time"

    input_path = os.path.abspath(os.path.join("static", "songs", filename))

    if not os.path.exists(input_path):
        return "Song file not found"

    os.makedirs("static/karaoke_clips", exist_ok=True)
    os.makedirs("static/generated/bgm", exist_ok=True)
    os.makedirs("static/generated/vocals", exist_ok=True)
    os.makedirs("static/lyrics", exist_ok=True)

    song_audio = AudioSegment.from_file(input_path)

    start_ms = start_sec * 1000
    end_ms = end_sec * 1000

    selected_clip = song_audio[start_ms:end_ms]

    if len(selected_clip) <= 0:
        return "Invalid preview selection"

  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_no_ext = os.path.splitext(filename)[0]

    clip_filename = "clip_" + name_no_ext + "_" + timestamp + ".mp3"
    clip_path = os.path.abspath(os.path.join("static", "karaoke_clips", clip_filename))

    selected_clip.export(clip_path, format="mp3", bitrate="192k")

    search_folder = os.path.abspath("temp_demucs")

    if os.path.exists(search_folder):
        shutil.rmtree(search_folder)

    output_folder = os.path.abspath("output")
    os.makedirs(output_folder, exist_ok=True)

    spleeter_python = sys.executable
    spleeter_fix_path = os.path.abspath("spleeter_fix.py")

    result = subprocess.run(
        [
      
            spleeter_python,
            spleeter_fix_path,
            "separate",
            clip_path,
            "-p",
            "spleeter:2stems",
            "-o",
            output_folder
        ],
  
        capture_output=True,
        text=True
    )

    bgm_path = None
    vocals_path = None

    for root, dirs, files in os.walk(search_folder):
        if "no_vocals.mp3" in files:
            bgm_path = os.path.join(root, "no_vocals.mp3")

        if "vocals.mp3" in files:
            vocals_path = os.path.join(root, "vocals.mp3")

        if bgm_path and vocals_path:
            break

    if bgm_path is None:
        return """
BGM generation failed
<br><br>
<b>STDOUT:</b><pre>{}</pre>
<b>STDERR:</b><pre>{}</pre>
""".format(result.stdout, result.stderr)

    final_filename = "bgm_" + name_no_ext + "_" + timestamp + ".mp3"
    final_path = os.path.abspath(
        os.path.join("static", "generated", "bgm", final_filename)
    )

    shutil.move(bgm_path, final_path)

    vocals_filename = "vocals_" + name_no_ext + "_" + timestamp + ".mp3"
    vocals_final_path = os.path.abspath(
       
        os.path.join("static", "generated", "vocals", vocals_filename)
    )

    if vocals_path and os.path.exists(vocals_path):
        shutil.move(vocals_path, vocals_final_path)

        vocals_audio = AudioSegment.from_file(vocals_final_path)
        vocals_audio = vocals_audio.normalize()
        vocals_audio.export(vocals_final_path, format="mp3")
    else:
        vocals_filename = clip_filename
        fallback_vocals_path = os.path.abspath(
            os.path.join("static", "generated", "vocals", vocals_filename)
        
        )
        shutil.copy(clip_path, fallback_vocals_path)
        vocals_final_path = fallback_vocals_path

    selected_song["karaoke_bgm"] = final_filename
    selected_song["karaoke_vocals"] = vocals_filename
    selected_song["karaoke_start"] = str(start_sec)
    selected_song["karaoke_end"] = str(end_sec)

    # ---------- LOAD ADMIN LYRICS ----------
    lyrics_folder = os.path.join("static", "lyrics")
    selected_song["lyrics"] = ""

    song_name_clean = clean_lyrics_name(selected_song.get("song_name", ""))

    for file in os.listdir(lyrics_folder):
        if not file.lower().endswith((".txt", ".lrc")):
          
            continue

        file_clean = clean_lyrics_name(os.path.splitext(file)[0])

        if (
            file_clean == song_name_clean
            or file_clean in song_name_clean
            or song_name_clean in file_clean
        ):
            lyrics_path = os.path.join(lyrics_folder, file)

            
            with open(lyrics_path, "r", encoding="utf-8") as f:
                lyrics_text = f.read()

            if file.lower().endswith(".lrc"):

                print("LRC FILE FOUND")

                selected_song["lyrics"] = convert_lrc_to_seconds(
                    lyrics_text,
        
                    start_sec,
                    end_sec
                )

                print(selected_song["lyrics"])

            else:

                preview_duration = end_sec - start_sec

    
                selected_song["lyrics"] = convert_txt_to_karaoke(
                    lyrics_text,
                    preview_duration
                )

                print(selected_song["lyrics"])

        # REMOVE INTRO / MUSIC LINES
         
                lyrics_text = "\n".join(
                    line for line in lyrics_text.splitlines()
                    if not (
                        "intro" in line.lower()
               
                        or "music" in line.lower()
                        or "instrumental" in line.lower()
                        or "bgm" in line.lower()
                    )   
           
                )

                selected_song["lyrics"] = convert_txt_to_karaoke(
                    lyrics_text,
                    preview_duration
                )

            break

    # ---------- ENGLISH AI FALLBACK ----------
    song_language = selected_song.get("language", "").strip().lower()

    if not selected_song.get("lyrics") and song_language in ["english", "en"]:
        try:
            import whisper

            model = whisper.load_model("base")

            result = model.transcribe(
                vocals_final_path,
                task="transcribe",
 
                fp16=False
            )

            auto_lines = []

            for segment in result["segments"]:
                start = round(segment["start"], 2)
                text = segment["text"].strip()

       
                if text:
                    auto_lines.append(f"{start}|{text}")

            if auto_lines:
                selected_song["lyrics"] = "\n".join(auto_lines)

        except Exception as e:
            print("Auto lyrics failed:", e)

    print("FINAL LYRICS:", selected_song.get("lyrics", "")[:200])

    save_songs_csv()

  
    return redirect(
        "/karaoke/" + filename +
        "?bgm_file=" + final_filename +
        "&vocals_file=" + vocals_filename +
        "&start=" + str(start_sec) +
        "&end=" + str(end_sec)
    )



@app.route("/ai_transcribe_vocals/<vocals_file>", methods=["POST"])
def ai_transcribe_vocals(vocals_file):
    if not login_required():
        return redirect("/login")

    original_song = request.args.get("song")

    selected_song = next(
        (s for s in songs if s.get("filename") == original_song and user_can_access_song(s)),
        None
    )

    
    if not selected_song:
        return "Song not found"

    vocals_path = os.path.join("static", "generated", "vocals", vocals_file)

    if not os.path.exists(vocals_path):
        return "Vocals file not found"

    try:
        import whisper

        model = whisper.load_model("base")

        song_language = selected_song.get("language", "").strip().lower()

        if song_language in ["telugu", "te"]:
            return "Telugu AI lyrics is disabled. Please use admin uploaded lyrics."
            result = model.transcribe(
                vocals_path,
                language="te",
                task="transcribe",
                fp16=False
            )
  
        else:
            result = model.transcribe(
                vocals_path,
                task="transcribe",
                fp16=False
            )

        lines = []

        for segment in result["segments"]:
            start = int(segment["start"])
            text = segment["text"].strip()

            if text:
                lines.append(f"{start}|{text}")

        selected_song["lyrics"] = "\n".join(lines)
        save_songs_csv()

    except Exception as e:
        return f"AI lyrics error: {str(e)}"

    return redirect(
        "/karaoke/" + original_song +
        "?vocals_file=" + vocals_file
    )

@app.route("/save_recording", methods=["POST"])
def save_recording():
    if not login_required():
        return {"message": "Login required"}, 401

    if "recording" not in request.files:
        return {"message": "No recording found"}, 400

    recording = request.files["recording"]
    bgm_filename = request.form.get("song")

    if not bgm_filename:
        return {"message": "BGM file missing"}, 400

   
    os.makedirs("static/recordings", exist_ok=True)
    os.makedirs("static/generated/karaoke", exist_ok=True)

    voice_path = os.path.join("static", "recordings", "voice_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm")
    recording.save(voice_path)

    bgm_path = os.path.join("static", "generated", "bgm", bgm_filename)

    if not os.path.exists(bgm_path):
        bgm_path = os.path.join("static", "songs", bgm_filename)

    if not os.path.exists(bgm_path):
        return {"message": "BGM file not found: " + bgm_filename}, 404

    voice_audio = AudioSegment.from_file(voice_path)
    bgm_audio = AudioSegment.from_file(bgm_path)

    min_len = min(len(voice_audio), len(bgm_audio))

    final_audio = bgm_audio[:min_len].overlay(voice_audio[:min_len])

  
    final_filename = "karaoke_mix_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp3"
    final_path = os.path.join("static", "generated", "karaoke", final_filename)

    final_audio.export(final_path, format="mp3", bitrate="192k")

    return {
        "message": "Final Voice + BGM created successfully",
        "file_url": "/static/generated/karaoke/" + final_filename,
        "filename": final_filename
    }


def get_mashup_settings(mashup_type, mood1="", mood2=""):
    mashup_type = (mashup_type or "smooth").lower()
    combined_mood = (mood1 + " " + mood2).lower()

    if mashup_type == "smooth":
     
        return 12, 12, 8, 2.5

    elif mashup_type == "dj":
        return 8, 16, 6, 1.5

    elif mashup_type == "romantic":
        return 15, 14, 10, 4.0

    elif mashup_type == "party":
        return 8, 18, 8, 2.0

    elif mashup_type == "chill":
        return 14, 12, 10, 4.5

    elif mashup_type == "auto":
        if "romantic" in combined_mood or "calm" in combined_mood or "sad" in combined_mood:
            return 15, 14, 10, 4.0

        elif "energetic" in combined_mood or "happy" in combined_mood or "party" in combined_mood:
            return 8, 18, 8, 2.0

        else:
            return 12, 12, 8, 2.5

    return 12, 12, 8, 2.5

@app.route("/mashup", methods=["GET", "POST"])

def mashup():
    if not login_required():
        return redirect("/login")

    result = None
    error = None
    compatibility = None

    if request.method == "POST":
        track_feature("Generate Mashup")
        increment_user_count(session.get("user"), "mashup_count")

        try:
            song1_file = request.form.get("song1")
            song2_file = request.form.get("song2")
           
            mashup_type = request.form.get("mashup_type", "smooth")
            selection_mode = request.form.get("selection_mode", "ai")

            visible_songs = get_visible_songs()

            song1 = next((s for s in visible_songs if s.get("filename") == song1_file), None)
            song2 = next((s for s in visible_songs if s.get("filename") == song2_file), None)

            if not song1 or not song2:
    
                error = "Invalid song selection"

            elif song1_file == song2_file:
                error = "Please select two different songs."

            else:
                if selection_mode == "custom":
                
                    compatibility = {
                        "emoji": "🎵",
                        "score": 100,
                        "level": "Custom Mashup",
                  
                        "reason": "Custom mode selected. Compatibility check skipped because user selected both songs manually.",
                        "can_mashup": True
                    }
                else:
                    compatibility = check_mashup_compatibility(song1, song2)

        
            if not compatibility["can_mashup"]:
                error = "Compatibility is low. Mashup cannot be generated."

            else:
                file1 = os.path.join("static", "songs", song1["filename"])
                file2 = os.path.join("static", "songs", song2["filename"])

                start1 = int(request.form.get("start1") or 0) * 1000
       
                start2 = int(request.form.get("start2") or 0) * 1000

                intro_sec, song2_sec, ending_sec, crossfade_sec = get_mashup_settings(
                    mashup_type,
                    song1.get("mood", ""),
       
                    song2.get("mood", "")
                )

                song1_intro = intro_sec * 1000
                song2_duration = song2_sec * 1000
              
                song1_ending = ending_sec * 1000
                crossfade = int(crossfade_sec * 1000)

                output = create_mashup(
                    file1,
                    file2,
                    start1,
                    start2,
                    song1_intro,
                    song2_duration,
                    song1_ending,
                    crossfade,
                    mashup_type
                )

        
            if output is None:
                error = "Song file missing. Check uploads."

            else:
                result = {
                    "song1": song1,
                        
                    "song2": song2,
                    "mashup_type": mashup_type.title(),
                    "mashup_name": song1.get("song_name", "Song 1") + " X " + song2.get("song_name", "Song 2"),
                           
                    "predicted_genre": song1.get("genre", "Unknown") + " + " + song2.get("genre", "Unknown"),                            
                    "predicted_mood": predict_mashup_mood(
                        song1.get("mood", "Unknown"),
                        song2.get("mood", "Unknown")
                    ),
                    "output": output,
          
                    "download_url": "/download_generated/mashups/" + os.path.basename(output)
                }

        except Exception as e:
            error = str(e)

    return render_template(
        "mashup.html",
        songs=get_visible_songs(),
        result=result,
 
        error=error,
        compatibility=compatibility
    )


@app.route("/download_generated/<folder>/<filename>")
def download_generated(folder, filename):
    if not login_required():
        return redirect("/login")

    allowed_folders = ["mashups", "bgm", "karaoke"]

    if folder not in allowed_folders:
        return "Invalid folder"

    file_path = os.path.join("static", "generated", folder, filename)

    if not os.path.exists(file_path):
        return "File not found"

    increment_user_count(session.get("user"), "download_count")
    track_feature("Download " + folder)

    return send_file(file_path, as_attachment=True)


@app.route("/delete_generated/<folder>/<filename>")
def delete_generated(folder, filename):
    global songs

    if not login_required():
        return redirect("/login")

    allowed_folders = ["mashups", "bgm", "karaoke"]

    if folder not in allowed_folders:
        return "Invalid folder"

    file_path = os.path.join("static", "generated", folder, filename)

    if os.path.exists(file_path):
        os.remove(file_path)

    if folder == "bgm":
        songs_copy = os.path.join("static", "songs", filename)

     
    if os.path.exists(songs_copy):
        os.remove(songs_copy)

    songs = [song for song in songs if song.get("filename") != filename]
    save_songs_csv()

    return redirect(request.referrer or "/home")

@app.route("/delete_song")
def delete_song():
    global songs

    if not login_required():
        return redirect("/login")

    filename = request.args.get("filename")

    if not filename:
        return "No filename provided"

    selected_song = next(
 
        (song for song in songs if song.get("filename") == filename),
        None
    )

    if not selected_song:
        return "Song not found"

    if not session.get("admin") and selected_song.get("uploaded_by") != session.get("user"):
        return "You cannot delete this file"

    song_path = os.path.join("static", "songs", filename)

    if os.path.exists(song_path):
        os.remove(song_path)

    songs = [song for song in songs if song.get("filename") != filename]
    save_songs_csv()

    return redirect(request.referrer or "/home")


@app.route("/download_song/<filename>")
def download_song(filename):
    if not login_required():
        return redirect("/login")

    selected_song = next(
        (song for song in songs if song.get("filename") == filename and user_can_access_song(song)),
        None
    )

    if not selected_song:
        return "You cannot access this file"

    song_path = os.path.join("static", "songs", filename)
    bgm_path = os.path.join("static", "generated", "bgm", filename)

    increment_user_count(session.get("user"), "download_count")
    track_feature("Download Song")

    if os.path.exists(song_path):
        return send_file(song_path, as_attachment=True)

    if os.path.exists(bgm_path):
        return send_file(bgm_path, as_attachment=True)

    return "File not found"

@app.route("/delete_mashup")
def delete_mashup():

    if not login_required():
        return redirect("/login")

    file = request.args.get("file")

    if file:
        file_path = file.replace("/", os.sep)

        if os.path.exists(file_path):
     
            os.remove(file_path)

    return redirect("/mashup")

@app.route("/edit_song/<filename>", methods=["POST"])
def edit_song(filename):
    if not login_required():
        return redirect("/login")

    selected_song = next(
        (song for song in songs if song.get("filename") == filename),
        None
    )

    if not selected_song:
        return "Song not found"

    if not session.get("admin") and selected_song.get("uploaded_by") != session.get("user"):
        return "You cannot edit this file"

    new_name = request.form.get("song_name")
    new_singer = request.form.get("singer")
    new_language = request.form.get("language")
    new_genre = request.form.get("genre")
    new_mood = request.form.get("mood")

    selected_song["song_name"] = new_name or selected_song.get("song_name", "")
    selected_song["singer"] = new_singer or selected_song.get("singer", "")
    selected_song["language"] = new_language or selected_song.get("language", "")
    selected_song["genre"] = new_genre or selected_song.get("genre", "")
    selected_song["mood"] = new_mood or selected_song.get("mood", "")

    save_songs_csv()

    return redirect(request.referrer or "/home")


@app.route("/admin-upload-lyrics", methods=["GET", "POST"])
def admin_upload_lyrics():
    if not session.get("admin"):
  
        return redirect("/admin-login")

    lyrics_folder = os.path.join("static", "lyrics")
    os.makedirs(lyrics_folder, exist_ok=True)

    if request.method == "POST":
        lyrics_files = request.files.getlist("lyrics_files")

        for lyrics_file in lyrics_files:
            if lyrics_file.filename == "":
                continue

            original_filename = lyrics_file.filename.lower()

        
            original_filename = original_filename.replace(".txt", "")
            original_filename = original_filename.replace(".lrc", "")

            save_filename = clean_lyrics_name(original_filename) + ".txt"
            save_path = os.path.join(lyrics_folder, save_filename)

            lyrics_file.save(save_path)

        return redirect("/admin-upload-lyrics")

    lyrics_files = [
        file for file in os.listdir(lyrics_folder)
        if file.lower().endswith((".txt", ".lrc"))
    ]

    return render_template(
        "admin_lyrics.html",
        lyrics_files=lyrics_files
    )

@app.route("/delete-lyrics/<filename>")
def delete_lyrics(filename):
    if not session.get("admin"):
        return redirect("/admin-login")

    lyrics_path = os.path.join("static", "lyrics", filename)

    if os.path.exists(lyrics_path):
        os.remove(lyrics_path)

    return redirect("/admin-upload-lyrics")


@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get("user"):
        return redirect("/login")

    message = request.form.get("message", "").strip()

   
    if message:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (username, message, created_at) VALUES (?, ?, ?)",
            (session.get("user"), message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()

        track_feature("Feedback")

  
    return redirect("/home")

@app.route("/ai_transcribe_clip/<filename>", methods=["POST"])
def ai_transcribe_clip(filename):
    if not login_required():
        return redirect("/login")

    selected_song = next(
        (s for s in songs if s.get("filename") == filename and user_can_access_song(s)),
        None
    )

    if not selected_song:
        return "You cannot access this file"

    start_sec = int(selected_song.get("karaoke_start") or 0)
    end_sec = int(selected_song.get("karaoke_end") or 30)

    song_path = os.path.join("static", "songs", filename)

 
    if not os.path.exists(song_path):
        return "Song file not found"

    song_audio = AudioSegment.from_file(song_path)
    clip = song_audio[start_sec * 1000:end_sec * 1000]

    lyrics_clip = os.path.join(
        "static",
        "karaoke_clips",
        "lyrics_clip_" + os.path.splitext(filename)[0] + ".mp3"
    )

    clip.export(lyrics_clip, format="mp3", bitrate="192k")

    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(lyrics_clip)

        lines = []

        for segment in result["segments"]:
            start = int(segment["start"])
            text = segment["text"].strip()

            if text:
                lines.append(f"{start}|{text}")

        selected_song["lyrics"] = "\n".join(lines)
  
        save_songs_csv()

    except Exception as e:
        return "Error during AI transcription: " + str(e)

    return redirect("/karaoke/" + filename)

@app.route("/admin-lyrics")
def admin_lyrics():
    if not session.get("admin"):
        return redirect("/admin-login")

    lyrics_folder = os.path.join("static", "lyrics")
    os.makedirs(lyrics_folder, exist_ok=True)

    lyrics_files = []

    for file in os.listdir(lyrics_folder):
        if file.lower().endswith((".txt", ".lrc")):
            lyrics_files.append(file)

  
    return render_template("admin_lyrics.html", lyrics_files=lyrics_files)

@app.route("/update_song/<filename>", methods=["POST"])
def update_song(filename):
    if not session.get("user"):
        return redirect("/login")
        
    username = session.get("user")
    new_genre = request.form.get("genre")