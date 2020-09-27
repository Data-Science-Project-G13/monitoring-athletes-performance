# Monitoring Athletes Performance
A CoachingMate analytical module to generate and deliver personalized feedback to coaches and athletes. Training sessions' descriptions could be 
merged with Garmin data to:
 - Assess if athletes are training as they should (Is the athlete training as suggested?).
 - Assess if athletes are training more or less than what they should (Is the athlete over-training?).
 - Predict/anticipate athlete risk fatigue levels (to avoid injuries).
 - Predict performance (If the athlete keep training in a particular way, how his/her fitness and form levels will increase?
   Can this athlete make a podium in a particular event in 6 months time, for example?)
 - Look at race performance data to compare predictabilities. 


## Guidelines

### Prerequisites

#### Python Version
```
Python 3.6
```

#### Envoirment Setup
1. Open the project `monitoring-athletes-performance` with IDE.
2. Use `the monitoring-athletes-performance/main` as the content root.
3. Setup Python Interpreter. Virtual environment is suggested.
4. Run the following commannd to install the required packages.
    ```
    pip3 install -r requirements.txt
    ```
5. Put the spreadsheet data (eg. `Eduardo Oliveira.csv`) in the `data` directory.
6. Put the folder (eg. `fit_eduardo_oliveira`) contains `.fit` files in the `data` directory.

What's more, our system supports you to simply convert `.fit` files.
Similarly, put the folder contains `.fit` files in the `data` directory, 
for example athlete `fit_eduardo_oliveira`, `fit_xu_chen` and `fit_carly_hart`, 
in your Terminal go to the `main` directory, and run the following commands. 
Folders contains converted .csv files will show up in `data` directory.
```
python3 fit_file_convert/process_all.py --subject-dir=../data --subject-name=csv_eduardo_oliveira --fit-source=../data/fit_eduardo_oliveira
python3 fit_file_convert/process_all.py --subject-dir=../data --subject-name=csv_xu_chen --fit-source=../data/fit_xu_chen
python3 fit_file_convert/process_all.py --subject-dir=../data --subject-name=csv_carly_hart --fit-source=../data/fit_carly_hart
```


