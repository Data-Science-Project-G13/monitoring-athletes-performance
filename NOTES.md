# Report Notes

## Data Cleaninng

### Original
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

## Building Models
