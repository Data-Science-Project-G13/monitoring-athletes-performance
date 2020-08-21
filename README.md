# Monitoring Athletes Performance
A CoachingMate analytical module to generate and deliver personalized feedback to coaches and athletes. Training sessions' descriptions could be 
merged with Garmin data to:
 - Assess if athletes are training as they should (is the athlete training as suggested?);
 - Assess if athletes are training more or less than what they should (is the athlete over-training?);
 - Predict/anticipate athlete risk fatigue levels (to avoid injuries)
 - Predict performance (if the athlete keep training in a particular way, how his/her fitness and form levels will increase?
 can this athlete make a podium in a particular event in 6 months time, for example?)
- Look at race performance data to compare predictabilities. 


## Guidelines

### Prerequisites

#### Using Docker
Install [Docker](https://docs.docker.com/get-docker/).

#### Using without Docker
```
pip install -r requirements.txt
```

### Converting .fit files to .csv files
Put the folder contains .fit files in the `data` directory, for example `fit_eduardo_oliveira`, then run the following
command in `main` module. A folder contains converted .csv files will show up in `data` directory.
```
python3 fit_file_convert/process_all.py --subject-dir=../data --subject-name=csv_eduardo_oliveira --fit-source=../data/fit_eduardo_oliveira
```


