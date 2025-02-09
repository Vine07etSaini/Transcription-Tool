import os
import argparse
import json
import whisper


def find_media_files(root_dir):
    """Recursively find audio/video files in directory"""
    media_extensions = [
        # Audio
        '.mp3', '.wav', '.m4a', '.flac', '.ogg',
        # Video
        '.mp4', '.mov', '.avi', '.mkv', '.webm'
    ]
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in media_extensions):
                yield os.path.join(root, file)

def main():
    parser = argparse.ArgumentParser(description='Media file transcription using Whisper')
    parser.add_argument('input_dir', help='Directory to process')
    parser.add_argument('--output_dir', help='Output directory (default: same as input files)')
    parser.add_argument('--format', choices=['txt', 'json'], default='txt',
                       help='Output format (default: txt)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing transcriptions')
    args = parser.parse_args()

    # Load Whisper model
    model = whisper.load_model("tiny", device="cpu")
    print("Loaded Whisper model")
    # Process files
    for media_path in find_media_files(args.input_dir):
        # Determine output path
        if args.output_dir:
            rel_path = os.path.relpath(media_path, args.input_dir)
            base_name = os.path.splitext(rel_path)[0]
            output_path = os.path.join(args.output_dir, base_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = os.path.splitext(media_path)[0]

        # Add format extension
        output_file = f"{output_path}_transcript.{args.format}"

        # Skip existing files unless overwrite is enabled
        if not args.overwrite and os.path.exists(output_file):
            print(f"Skipping existing transcript: {output_file}")
            continue

        # Transcribe media file
        print(f"Processing: {media_path}")
        try:
            result = model.transcribe(media_path,fp16=False)
        except Exception as e:
            print(f"Error processing {media_path}: {str(e)}")
            continue

        # Save results
        if args.format == 'txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])
        else:  # json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Created transcript: {output_file}")

if __name__ == "__main__":
    main()