# Converting Garmin FIT to CSV (and GPX to CSV)

[article for original version](https://maxcandocia.com/article/2017/Sep/22/converting-garmin-fit-to-csv/)

Requirements:

 * Python 3.5+

 * BeautifulSoup

 * fitparse (installation instructions below)

 * tzwhere (to localize timezones)

 * pytz (to localize timezones)

 * FIT files to convert (if using that feature)

 * GPX files to convert (if using that feature)

First, install `fitparse`

    sudo pip3 install -e git+https://github.com/dtcooper/python-fitparse#egg=python-fitparse

OR

    sudo pip3 install fitparse

Then you can execute `process_all.py`

    python3 process_all.py --subject-name=mysubjectname --fit-source-dir=/media/myname/GARMIN/Garmin/ACTIVITY/

This will create a bunch of CSVs for all of your workouts in that directory. The files will be stored in a `subject_data` directory's subdirectory (check the defaults for the specific folders), which is generated based off the subject name. Up to 3 files are made per FIT file:

 1. A CSV of all of the track data

 2. A CSV of the lap data

 3. A CSV of the start (and stop) data

Each of the CSVs is in the format '{activity_type}_YY-MM-DD_HH-MM-SS[_{laps,starts}].csv.

In terms of this monitoring-athletes-performance project, go to directory `main` and run

     python3 fit_file_convert/process_all.py --subject-dir=../data --subject-name=csv_eduardo_oliveira --fit-source=../data/fit_eduardo_oliveira

You can also provide a csv to censor certain geographic regions by latitude, longitude, and radius. Simply create a CSV with `longitude`, `latitude`, and `radius` column headers, and add as many circular regions as you want. Note that radius is assumed to be in meters.
    
    python3 process_all.py --subject-name=mysubjectname --fit-source-dir=/media/myname/GARMIN/Garmin/ACTIVITY/ --censorfile=/home/mydir/censor.csv --censor

This will be stored in a folder called `"censored"` that is in the subject's directory. You can use the `--censor-string=` to change what censored fields are replaced with (default is `[CENSORED]`).

You can also archive the data after it's been processed:

    python3 process_all.py --subject-name=mysubjectname --fit-source-dir=/media/myname/GARMIN/Garmin/ACTIVITY/ --censorfile=/home/mydir/censor.csv --censor --archive-results

By default, this stores data in a directory called `archives` in the main `subject_data` folder. You can add the `--archive-censored-only`, which will only archive the censored folder.

## GPX data

You can also process GPX data (and censor it the same way as FIT data)

For the initial processing, you can do

    python3 process_all.py --subject-name=mysubjectname --skip-fit-conversion gpx-source-dir=/home/mydir/gpx_files

By default, the program will always try to copy/process FIT files unless you add the `--skip-fit-conversion` flag, but you can always tweak the code to your needs.

## Additional Help

You can use `python3 process_all.py --help` to see more information.# fit-file-processing

```angular2
optional arguments:
  -h, --help            show this help message and exit
  --subject-name SUBJECT_NAME
                        name of subject
  --fit-source-dir FIT_SOURCE_DIR
                        source data for garmin fit
  --fit-target-dir FIT_TARGET_DIR
                        target directory for FIT data; default uses subject
                        name
  --fit-processed-csv-dir FIT_PROCESSED_CSV_DIR
                        target directory for CSVs of processed fit data;
                        default uses subject name
  --erase-copied-fit-files
                        If True, will delete any copied FIT files (not the
                        originals, though)
  --gpx-source-dir GPX_SOURCE_DIR
                        directory for gpx files (if desired)
  --gpx-target-dir GPX_TARGET_DIR
                        directory to store processed gpx csv in
  --subject-dir SUBJECT_DIR
                        default directory to store subject data in
  --gpx-summary-filename GPX_SUMMARY_FILENAME
                        the summary filename for gpx data
  --fit-overwrite       Will overwrite any previously created CSVs from fit
                        data
  --fit-ignore-splits-and-laps
                        Will not write split/lap data if specified
  --censorfile CENSORFILE
                        If provided, will use censorfile CSV to create a copy
                        of data with censored locations around different
                        latitude/longitude/radii
  --censor-string CENSOR_STRING
                        This is what censored fields are replaced with in
                        censored data
  --archive-results     If True, will package data into an archive
  --archive-censored-only
                        If True, will only package data that is censored
  --archive-extra-files ARCHIVE_EXTRA_FILES [ARCHIVE_EXTRA_FILES ...]
                        Will copy these extra files into an archive if it is
                        being created
  --archive-output-dir ARCHIVE_OUTPUT_DIR
                        location for archived output
  --archive-filename ARCHIVE_FILENAME
                        archive filename; will use name for default if none
                        specified
  --skip-gpx-conversion
                        Skips GPX conversion if used
  --skip-fit-conversion
                        Skips FIT conversion if used

```