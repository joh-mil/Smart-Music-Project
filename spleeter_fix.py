import sys
import os
import subprocess
import shutil

def main():
    # Spleeter CLI: separate <input_file> -p spleeter:2stems -o <output_dir> -B tensorflow
    # We want to emulate this using Demucs.
    
    if len(sys.argv) < 2:
        print("Usage: python spleeter_fix.py separate <input_file> ...")
        return

    command = sys.argv[1]
    if command != "separate":
        print(f"Command {command} not supported by fix script.")
        return

    # Add the local bin directory to PATH so FFmpeg can be found
    bin_dir = os.path.abspath(os.path.join(os.getcwd(), "bin"))
    ffmpeg_path = os.path.join(bin_dir, "ffmpeg.exe")
    if os.path.exists(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

    input_file = None
    output_dir = "output"
    
    # Simple argument parser
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-o":
            output_dir = sys.argv[i+1]
            i += 2
        elif arg == "-p":
            i += 2
        elif arg == "-B":
            i += 2
        elif not arg.startswith("-"):
            input_file = arg
            i += 1
        else:
            i += 1
            
    if not input_file:
        print("No input file provided.")
        return
    
    # CONVERSION STEP: Convert to WAV first to avoid torchaudio/torchcodec DLL issues
    print(f"--- Spleeter Fix: Converting {input_file} to WAV ---")
    wav_file = os.path.splitext(input_file)[0] + "_temp.wav"
    
    # Remove old temp wav if exists
    if os.path.exists(wav_file):
        os.remove(wav_file)
        
    conv_cmd = [ffmpeg_path, "-i", input_file, wav_file]
    conv_result = subprocess.run(conv_cmd, capture_output=True, text=True)
    
    if conv_result.returncode != 0:
        print("FFmpeg conversion failed!")
        print(conv_result.stderr)
        # Fallback to original file if conversion failed, but it will likely fail in demucs too
    else:
        input_file = wav_file

    print(f"--- Spleeter Fix: Using Demucs to process {input_file} ---")
    
    # Run Demucs
output_dir = os.path.join(os.getcwd(), "temp_demucs")
os.makedirs(output_dir, exist_ok=True)

input_file = sys.argv[2]

vpython = sys.executable

demucs_cmd = [
    sys.executable,
    "-m", "demucs",
    "--two-stems", "vocals",
    "--mp3",
    "--mp3-bitrate", "192",
    "-o", output_dir,
    input_file
]

print(f"Running: {' '.join(demucs_cmd)}")

result = subprocess.run(
    demucs_cmd,
    capture_output=True,
    text=True
)
    
    # Clean up temp wav
print("Demucs completed successfully!")

if result.returncode != 0:
    print("Demucs failed!")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)

print("Demucs completed successfully!")
    # Demucs output structure: temp_demucs/htdemucs/<filename>/vocals.wav and no_vocals.wav
filename = os.path.splitext(os.path.basename(input_file))[0]
    # If we used temp wav, the filename might have _temp
if filename.endswith("_temp"):
        filename = filename[:-5]
        
src_dir = os.path.join("temp_demucs", "htdemucs", os.path.splitext(os.path.basename(input_file))[0])
    
    # Spleeter structure: <output_dir>/<filename>/vocals.wav and accompaniment.wav
    # The original filename without _temp
original_filename = os.path.splitext(os.path.basename(sys.argv[2]))[0]
dst_dir = os.path.join(output_dir, original_filename)
if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        
print(f"Moving results to {dst_dir}")
    
    # Copy vocals
if os.path.exists(os.path.join(src_dir, "vocals.mp3")):
    shutil.copy(
        os.path.join(src_dir, "vocals.mp3"),
        os.path.join(dst_dir, "vocals.mp3")
    )

if os.path.exists(os.path.join(src_dir, "no_vocals.mp3")):
    shutil.copy(
        os.path.join(src_dir, "no_vocals.mp3"),
        os.path.join(dst_dir, "no_vocals.mp3")
    )

print("Files copied successfully!")
    
print("--- Spleeter Fix: Success! ---")

if __name__ == "__main__":
    main()
