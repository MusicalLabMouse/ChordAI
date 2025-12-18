"""
Find close matches for unmatched songs in the CSV.
"""

import csv
import re
from pathlib import Path
from difflib import SequenceMatcher


def normalize_text(text):
    """Normalize text for matching."""
    text = text.lower()
    text = re.sub(r'\.(mp3|wav|mp4)$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[_\-/\\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_song_from_filename(filename):
    """Extract song name from filename."""
    name = Path(filename).stem
    if ' - ' in name:
        parts = name.split(' - ')
        song_name = parts[0].strip()
    else:
        song_name = name
    return song_name


def similarity(a, b):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def find_close_matches(unmatched_files, csv_path, threshold=0.6):
    """
    Find close matches for unmatched files.

    Args:
        unmatched_files: List of unmatched filenames
        csv_path: Path to CSV file
        threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        Dictionary mapping filename to list of potential matches
    """
    # Load CSV data
    csv_songs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_songs.append({
                'song_id': row['song_id'],
                'song_name': row['song_name'],
                'artist': row['artist'],
                'normalized': normalize_text(row['song_name'])
            })

    results = {}

    for filename in unmatched_files:
        song_name = extract_song_from_filename(filename)
        normalized_song = normalize_text(song_name)

        # Find potential matches
        matches = []
        for csv_song in csv_songs:
            score = similarity(normalized_song, csv_song['normalized'])
            if score >= threshold:
                matches.append({
                    'song_id': csv_song['song_id'],
                    'song_name': csv_song['song_name'],
                    'artist': csv_song['artist'],
                    'similarity': score
                })

        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        results[filename] = {
            'extracted_song': song_name,
            'matches': matches[:5]  # Top 5 matches
        }

    return results


def main():
    # List of unmatched files
    unmatched_files = [
        "A Love Song - Live - Anne Murray.mp3",
        "After The Love Is Gone - Earth, Wind & Fire.mp3",
        "Big Yellow Taxi (2007) - Joni Mitchell.mp3",
        "Chicago We Can Change the World - 2018 Remaster - Graham Nash.mp3",
        "Crystal Blue Persuasions (Single Version) - Tommy James & The Shondells.mp3",
        "Doo Doo Doo Doo (Heartbreaker) - 2005 Digital Remaster - The Rolling Stones.mp3",
        "El Condor Pasa (If I Could) - Daniel Alomía Robles, Simon & Garfunkel.mp3",
        "Get Up I Feel Like Being Like A Sex Machine, Pts. 1 & 2 - James Brown, The Original J.B.s.mp3",
        "I Got You - 1964 Smash Version - James Brown.mp3",
        "I've Been Loving You Too Long - To Stop Now - Otis Redding.mp3",
        "Indiana Wants Me (Rerecorded) - R. Dean Taylor.mp3",
        "Kicks (feat. Mark Lindsay) - Single Version - Paul Revere & The Raiders, Mark Lindsay.mp3",
        "Living It Down ( [en espanol]) - Freddy Fender.mp3",
        "Make Me Smile Now More Than Ever - 2002 Remaster - Chicago.mp3",
        "Old Time Rock And Roll - Remastered 2011 - Bob Seger.mp3",
        "Reelin' And Rockin' - Chuck Berry.mp3",
        "Running Up That Hill (A Deal With God) - 2018 Remaster - Kate Bush.mp3",
        "Somebody's Watching Me - Single Version - Rockwell.mp3",
        "That's Old Fashioned (The Way Love Should Be) - The Everly Brothers.mp3",
        "The Anaheim, Azusa And Cucamonga Sewing Circle, Book Review And Timing Association - Jan & Dean.mp3",
        "The Best of My Love - 2013 Remaster - Eagles.mp3",
        "The Sound of Silence - Electric Version - Simon & Garfunkel.mp3",
        "Walk Like A Man (You Can Call Me Your Man) - Remastered 1999 - Grand Funk Railroad.mp3",
        "Wendy (Stereo) - The Beach Boys.mp3",
        "Willie And The Hand Jive Get Ready - Live At Long Beach Arena, California 1974 - Eric Clapton.mp3",
        "You Can't Judge A Book By Its Cover - Bo Diddley.mp3",
        "You've Got Another Thing Coming - Judas Priest.mp3"
    ]

    csv_path = '../mcgill_billboard_spotify.csv'

    print("=" * 80)
    print("CLOSE MATCH ANALYSIS FOR UNMATCHED SONGS")
    print("=" * 80)

    results = find_close_matches(unmatched_files, csv_path, threshold=0.5)

    matched_count = 0
    likely_matches = []

    for filename, data in results.items():
        extracted = data['extracted_song']
        matches = data['matches']

        print(f"\n{'-' * 80}")
        print(f"FILE: {filename}")
        print(f"Extracted Song Title: {extracted}")
        print(f"{'-' * 80}")

        if matches:
            print(f"Found {len(matches)} potential match(es):\n")
            for i, match in enumerate(matches, 1):
                score_pct = match['similarity'] * 100
                print(f"  [{i}] song_id={match['song_id']}: {match['song_name']} - {match['artist']}")
                print(f"      Similarity: {score_pct:.1f}%")

                if i == 1 and score_pct >= 70:
                    matched_count += 1
                    likely_matches.append({
                        'filename': filename,
                        'song_id': match['song_id'],
                        'csv_title': match['song_name'],
                        'artist': match['artist'],
                        'similarity': score_pct
                    })
        else:
            print("  No close matches found (threshold: 50%)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - LIKELY MATCHES (Similarity >= 70%)")
    print("=" * 80)

    if likely_matches:
        print(f"\nFound {len(likely_matches)} likely matches:\n")
        for match in likely_matches:
            print(f"MP3: {match['filename']}")
            print(f"  => song_id={match['song_id']}: {match['csv_title']} - {match['artist']}")
            print(f"  Similarity: {match['similarity']:.1f}%")
            print()
    else:
        print("\nNo high-confidence matches found.")

    print(f"Total unmatched: {len(unmatched_files)}")
    print(f"Likely matches found: {len(likely_matches)}")
    print(f"Still unmatched: {len(unmatched_files) - len(likely_matches)}")

    # Save report
    report_path = 'close_matches_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("LIKELY MATCHES FOR UNMATCHED SONGS\n")
        f.write("=" * 80 + "\n\n")

        for match in likely_matches:
            f.write(f"MP3 File: {match['filename']}\n")
            f.write(f"Assigned song_id: {match['song_id']}\n")
            f.write(f"CSV Title: {match['csv_title']}\n")
            f.write(f"Artist: {match['artist']}\n")
            f.write(f"Similarity: {match['similarity']:.1f}%\n")
            f.write("\n")

    print(f"\nReport saved to: {report_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
