from spleeter.separator import Separator
import os

def main():
    # Choose the model: 2stems, 4stems, or 5stems
    separator = Separator('spleeter:4stems')  # drums, bass, piano/other, vocals

    # Input audio file
    input_audio = r'C:\Users\User\Downloads\MIMI - Science.mp3'

    # Output directory
    output_dir = r'C:\Users\User\Desktop\Midivise\quality_tests\output'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run separation
    separator.separate_to_file(input_audio, output_dir)

    print("Separation complete!")

if __name__ == "__main__":
    main()
