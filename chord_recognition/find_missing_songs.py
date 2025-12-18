"""
Find which 2 songs from the songs folder don't have a folder in training_data.
"""

from pathlib import Path


def find_missing(songs_dir, training_dir):
    """Find songs that don't have corresponding folders."""

    # Manual mappings from both scripts
    all_manual_mappings = {
        # From add_manual_songs.py
        "Kicks (feat. Mark Lindsay) - Single Version - Paul Revere & The Raiders, Mark Lindsay.mp3": "1181",
        "Make Me Smile Now More Than Ever - 2002 Remaster - Chicago.mp3": "1085",
        "Walk Like A Man (You Can Call Me Your Man) - Remastered 1999 - Grand Funk Railroad.mp3": "0762",
        "Wendy (Stereo) - The Beach Boys.mp3": "0479",

        # From add_close_matches.py
        "A Love Song - Live - Anne Murray.mp3": "0917",
        "After The Love Is Gone - Earth, Wind & Fire.mp3": "1056",
        "Big Yellow Taxi (2007) - Joni Mitchell.mp3": "0668",
        "Chicago We Can Change the World - 2018 Remaster - Graham Nash.mp3": "0248",
        "Crystal Blue Persuasions (Single Version) - Tommy James & The Shondells.mp3": "0528",
        "Doo Doo Doo Doo (Heartbreaker) - 2005 Digital Remaster - The Rolling Stones.mp3": "0345",
        "El Condor Pasa (If I Could) - Daniel Alomía Robles, Simon & Garfunkel.mp3": "1261",
        "Get Up I Feel Like Being Like A Sex Machine, Pts. 1 & 2 - James Brown, The Original J.B.s.mp3": "0857",
        "I Got You - 1964 Smash Version - James Brown.mp3": "0914",
        "I've Been Loving You Too Long - To Stop Now - Otis Redding.mp3": "0502",
        "Indiana Wants Me (Rerecorded) - R. Dean Taylor.mp3": "0030",
        "Living It Down ( [en espanol]) - Freddy Fender.mp3": "0346",
        "Old Time Rock And Roll - Remastered 2011 - Bob Seger.mp3": "0249",
        "Reelin' And Rockin' - Chuck Berry.mp3": "1110",
        "Running Up That Hill (A Deal With God) - 2018 Remaster - Kate Bush.mp3": "0531",
        "Somebody's Watching Me - Single Version - Rockwell.mp3": "0334",
        "That's Old Fashioned (The Way Love Should Be) - The Everly Brothers.mp3": "1248",
        "The Anaheim, Azusa And Cucamonga Sewing Circle, Book Review And Timing Association - Jan & Dean.mp3": "0689",
        "The Best of My Love - 2013 Remaster - Eagles.mp3": "1061",
        "The Sound of Silence - Electric Version - Simon & Garfunkel.mp3": "1153",
        "Willie And The Hand Jive Get Ready - Live At Long Beach Arena, California 1974 - Eric Clapton.mp3": "0194",
        "You Can't Judge A Book By Its Cover - Bo Diddley.mp3": "0179",
        "You've Got Another Thing Coming - Judas Priest.mp3": "0861"
    }

    training_path = Path(training_dir)
    existing_folders = set(f.name for f in training_path.iterdir() if f.is_dir())

    print("=" * 80)
    print("FINDING SONGS WITH DUPLICATE song_ids")
    print("=" * 80)

    duplicates_found = []

    for mp3_filename, song_id in all_manual_mappings.items():
        if song_id in existing_folders:
            # Check if this was added by organize_dataset or manual script
            # by checking the MP3 file inside the folder
            folder_path = training_path / song_id
            mp3_files = list(folder_path.glob('*.mp3'))

            if len(mp3_files) == 1:
                # Only one MP3, so this song_id existed before manual addition
                print(f"\nDUPLICATE song_id={song_id}")
                print(f"  Manual MP3: {mp3_filename}")
                print(f"  Existing MP3: {mp3_files[0].name}")
                duplicates_found.append(song_id)

    print("\n" + "=" * 80)
    print(f"Found {len(duplicates_found)} songs with duplicate song_ids")
    print("=" * 80)

    print(f"\nThese {len(duplicates_found)} MP3s from the songs folder could not be added")
    print("because their song_id was already assigned to a different song.")


if __name__ == '__main__':
    songs_dir = '../songs'
    training_dir = '../training_data'

    find_missing(songs_dir, training_dir)
