# Briefing:

## Assignment 1 (undergrad) CS 370 
## Date: 9/28/25


### Part A: Simulating Average NN Distance in an Increasing Dimension Space

### Part B: Making a Logistic Regression Model with SGD and Comparing Recall vs Precision. 

### Instructions: 
- I was unable to push the test.gz file for how large it was; add it to the assignment-1 folder if running code
- Launch the Docker and the VM
- Go to /assignments/assignment-1/01-assignment-part-1.ipynb for the annotated code, and outputs from the assignment



# In-Depth:

## Assignment 1: Logistic Regression Implementation
This assignment demonstrates the implementation of logistic regression using both scikit-learn and manual PyTorch implementations, applied to click-through rate (CTR) prediction.

### Part A: Nearest Neighbor Analysis in High Dimensions
#### Objective: Investigate how the distance to nearest neighbors changes as dimensionality increases, demonstrating the "curse of dimensionality" phenomenon.

#### Implementation:

- Generated 1,000 random points in dimensions ranging from 1 to 100
- Used PyTorch tensors for efficient computation
- Calculated Euclidean distances between all point pairs
- Computed average nearest neighbor distances for each dimensionality
#### Key Findings:

- As dimensionality increases, the average distance to nearest neighbors grows significantly
- This demonstrates the curse of dimensionality: in high-dimensional spaces, all points become approximately equidistant
- Results visualized on a log-scale plot showing the exponential growth pattern
#### Technical Details:

- Used torch.rand() for uniform random point generation
- Employed np.linalg.norm() for distance calculations
- Applied logarithmic scaling for clear visualization of the relationship
### Part B: Click-Through Rate Prediction
#### Objective: Implement and compare logistic regression models for binary classification of advertisement click-through rates using both scikit-learn and manual PyTorch implementations.

#### Data Processing
- Dataset: Large-scale CTR prediction dataset with 19 features
- Features: Hour, banner position, site/app information, device characteristics, and categorical variables (C1, C14-C21)
- Memory Management: PyArrow streaming for efficient processing of large compressed files
- Data Cleaning: Handled missing values, infinite values, and ensured numeric consistency
- Train/Test Split: 80/20 split with stratification to maintain class balance
- Scikit-learn Implementation
- Algorithm: SGDClassifier with logistic loss function
- Preprocessing: StandardScaler for feature normalization
#### Results:
- Test Accuracy: ~82.6%
- AUC Score: High performance on imbalanced dataset
- Precision-Recall Analysis: Significant improvement over random baseline
- Manual PyTorch Implementation
#### Core Functions:

- sigmoid(): Numerically stable activation function
- bce_loss(): Binary cross-entropy with L2 regularization
- grads(): Manual gradient computation for weights and bias
- train_sgd(): Stochastic gradient descent with mini-batches
#### Key Features:

- Data Standardization: Critical preprocessing step using SimpleImputer and StandardScaler to prevent NaN issues
- Gradient Descent: Both batch and stochastic implementations
- Regularization: L2 penalty to prevent overfitting
- Mini-batch Processing: Efficient SGD with shuffling and batch processing
#### Training Configuration:

- Learning Rate: 0.05 (optimized for SGD noise)
- Epochs: 30 (sufficient for convergence)
- Batch Size: 64 (balanced efficiency and gradient quality)
- Regularization: λ = 1e-3
#### Performance Analysis
- Precision-Recall Curve: Comprehensive analysis with optimal threshold detection
- F1-Score Optimization: Automated threshold tuning for balanced precision/recall
- Baseline Comparison: Demonstrated significant improvement over random classification
- Loss Tracking: Binary cross-entropy loss monitoring for both implementations
#### Technical Highlights
- Numerical Stability: Proper handling of infinite values and NaN prevention
- Memory Efficiency: Streaming data processing and garbage collection
- Reproducibility: Fixed random seeds for consistent results
- Error Handling: Robust preprocessing pipeline with comprehensive data validation
#### Code Organization
- Modular Design: Separate functions for each mathematical operation
- Comprehensive Documentation: Detailed comments explaining mathematical concepts and acronyms
- Educational Value: Step-by-step implementation showing the mathematics behind logistic regression
## Key Learning Outcomes
- High-Dimensional Geometry: Understanding how distance metrics behave in high-dimensional spaces
- Manual ML Implementation: Deep understanding of logistic regression mathematics and optimization
- Data Engineering: Large-scale data processing with memory-efficient streaming
- Model Evaluation: Comprehensive performance analysis using multiple metrics
- Numerical Computing: Handling numerical stability issues in machine learning implementations
## Technologies Used
- PyTorch: For tensor operations and manual gradient computation
- scikit-learn: For comparison implementation and preprocessing utilities
- PyArrow: For efficient large file processing
- NumPy/Pandas: For data manipulation and analysis
- Matplotlib: For visualization and results presentation

This implementation demonstrates both theoretical understanding and practical engineering skills in machine learning, showcasing the ability to implement algorithms from scratch while handling real-world data challenges.
