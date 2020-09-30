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

### How to Run Our System

#### User Option
Open the system built-in command-line tools or Terminal in a Python IDE, you can go to the main folder of the project,
and run the following command. 
```angular2
python system_argparse.py --athletes-names xu_chen eduardo_oliveira --initialize-system=False --clean-data=False --generate-pmc=True
```
The example above means you don't want to initialize the system (say it's your second time running this system),
but you do want Performance Management Charts for Xu Chen and Eduardo Oliveira (say you have updated the fit files).
Please refer to [Additional Help](#additional-help) for more options.

#### Developer Option
For the developers work further based on this system, we recommand changing options in `main.py` for test, and 
here's an overview.
```Python
if __name__ == '__main__':
    athletes_names = ['eduardo_oliveira', 'xu_chen', 'carly_hart']
    internal_args = ['--athletes-names={}'.format(' '.join(athletes_names)),
                     '--initialize-system=False', '--clean-data=False', '--process-feature-engineering=False',
                     '--build-model=False', '--generate-pmc=True']
    main(internal_args)
```


### Additional Help
```
optional arguments:
  -h, --help        show this help message and exit
  --initialize-system  INITIALIZE_SYSTEM
                    whether initialize the system, default is True  
  --clean-data  CLEAN_DATA
                    whether clean the data, default is True  
  --process-feature-engineering  FEATURE_ENGINEERING
                    whether do feature engineering, default is True  
  --build-model  BUILD_MODEL
                    whether process modeling, default is True  
  --generate-pmc  GENERATE_PMC
                    whether generate PMC, default is True  
  --athletes-names  ATHLETES_NAME
                    the athletes' names split by spaces
```