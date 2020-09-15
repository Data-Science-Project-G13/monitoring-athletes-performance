# Report Notes

## Data Cleaninng

### Spreadsheet
#### Problems and possible solutions


### Additional
#### Problems and solutions
- Values in a column are all missing
- Can't apply multivariate imputation when one of the column is missing
    - Solution1: Handle the whole missing column first
    - Solution2: Ignore it
    - Solution3: Use interpolation instead
    
- Series data outlier detection
    - Valid data but in small cluster
        - Solution1: Use regression to see the trend
    - Outlier which doesn't follow the trend but not able to detect by boxplot
        - Solution1: Check trend
    
    
## Feature Engineering
### Spreadsheet
### Additional 

### Merging
Use dictionary to lower the merging complexity from O(N^2) to O(1)*O(N)
The activity time on a certain date in additional data and in spreadsheet data
are not always match, hence match them by date and the activity types.


## Building Models
May need to train models activity-wisely, since the features are filling out differently between activities.
Otherwise, should only choose the features that have same situations across activities.

Because imputation was done activity-wisely, many of the values are nnot imputed.
