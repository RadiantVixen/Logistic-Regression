🪄 Harry Potter and the Data Scientist: Multi-Class Logistic Regression from Scratch
--Image of: --Python Version --Image of: --No Dependencies --Image of: --Hogwarts House Classification --Image of: --License: MIT

Welcome to Harry Potter and the Data Scientist, an elite, dependency-free implementation of multi-class classification using One-vs-All (OvA) Logistic Regression built completely from scratch.

This repository models the Sorting Hat's magic by classifying Hogwarts students into their respective houses (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) based on their academic scores across various magical subjects. It provides custom tools for exploratory data analysis, mathematical statistical summaries, feature standardization, and optimization via three variants of gradient descent.

🗺️ Project Pipeline Architecture
graph TD
    A[dataset_train.csv] -->|describe.py| B(Mathematical Summaries)
    A -->|histogram.py / scatter_plot.py / pair_plot.py| C(Data Visualizations)
    A -->|logreg_train.py| D{Optimization Engine}

    D -->|Batch GD| E[weights.txt]
    D -->|Stochastic GD| E
    D -->|Mini-Batch GD| E

    E -->|logreg_predict.py| F[dataset_test.csv]
    F --> G[houses.csv (Sorting Hat Predictions)]
The system is separated into three highly optimized decoupled modules:

The Profiler (describe.py): Computes statistical summaries (mean, std, min, percentiles, max, IQR) without relying on any external packages like Pandas.
The Visualizer (histogram.py, scatter_plot.py, pair_plot.py): Conducts EDA to identify key features.
The Classifier (logreg_train.py, logreg_predict.py): Recreates multi-class logistic regression with feature Z-score scaling.
🎨 Interactive Visual Showcases
1. Exploratory Data Analysis & Feature Footprints
Before training, we must select which features carry the actual house-specific footprints. This animation shows a Feature Morpher that interpolates distributions between noisy/overlapping features and highly separable ones. Notice how Arithmancy is completely homogeneous (useless), whereas Charms has perfectly distinct peaks for each house.

2. The Battle of Gradient Descents
This dashboard compares the three custom optimization algorithms in real-time on the loss landscape:

Batch Gradient Descent (Blue): Runs a smooth, direct path to the global minimum, maintaining maximum mathematical stability but requiring the full dataset per step.
Stochastic Gradient Descent (Red): Takes noisy, randomized, erratic steps but converges incredibly fast, mimicking high-frequency updates.
Mini-batch Gradient Descent (Purple): Strikes the ultimate balance, maintaining a clean, semi-smooth trajectory with high computational efficiency.
3. One-vs-All Decision Boundary Evolution
Watch how the four independent binary classifiers adjust their sigmoid decision boundaries ($P = 0.5$) in the 2D magical space of Astronomy vs. Herbology. As training epochs increment, the classification regions morph and lock in to cleanly isolate the four Hogwarts Houses.

🧮 Mathematical Foundations
1. The Sigmoid Function
To map any real-valued number into a probability $P \in (0, 1)$, we employ the Sigmoid activation function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

For numerical stability, we handle positive and negative inputs separately to avoid exponent overflow:

$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \geq 0 \ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

2. Cross-Entropy Loss (Log-Loss)
For each binary classification model, we optimize the weights and bias by minimizing the average negative log-likelihood:

$$J(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log(p^{(i)}) + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]$$

3. Gradient Updates
The weights $w$ and bias $b$ are adjusted at each step in the direction of steepest descent:

$$w_j \leftarrow w_j + \alpha \frac{1}{M} \sum_{k=1}^{M} \left( y^{(k)} - p^{(k)} \right) x_j^{(k)}$$

$$b \leftarrow b + \alpha \frac{1}{M} \sum_{k=1}^{M} \left( y^{(k)} - p^{(k)} \right)$$

where $\alpha$ is the learning rate, and $M$ is the update window ($M=N$ for Batch, $M=1$ for SGD, $M=\text{batch_size}$ for Mini-batch).

4. One-vs-All Classification Rule
Since we have 4 houses, we train 4 independent binary classifiers. Each house classifier outputs the probability that a student belongs to that specific house. The final predicted house is the one that yields the highest probability:

$$\hat{y} = \arg\max_{c \in {\text{Gryffindor, Hufflepuff, Ravenclaw, Slytherin}}} \sigma\left( w_c^T x + b_c \right)$$

📂 Repository Anatomy
├── describe.py               # Generates custom descriptive statistics (No Pandas!)
├── logreg_train.py           # Trains the OvA model using 3 different optimization methods
├── logreg_predict.py         # Loads weights.txt and performs inference on new datasets
├── evaluate.py               # Comparative benchmark harness of all three gradient descents
├── histogram.py              # Visualizes score distributions across houses
├── scatter_plot.py           # Visualizes relationships between feature pairs
├── pair_plot.py              # Large-scale multi-feature correlation matrix
├── weights.txt               # Serialized weights, biases, and normalization stats
└── utils/
    ├── csv_reader.py         # Custom dependency-free CSV parser
    ├── stats.py              # Custom math/statistical utility functions
    └── normalization.py     # Custom fitting & applying of Z-score normalization
🚀 Execution Guide
1. Mathematical Summarization
Generate standard descriptive statistics for any dataset:

python describe.py datasets/dataset_train.csv
2. Training the Model
Train the multi-class model with your preferred optimization algorithm:

# Train using Batch Gradient Descent (Default: lr=0.5, epochs=150)
python logreg_train.py datasets/dataset_train.csv -m batch

# Train using Stochastic Gradient Descent (lr=0.01, epochs=15)
python logreg_train.py datasets/dataset_train.csv -m sgd

# Train using Mini-batch Gradient Descent (lr=0.1, epochs=30, batch-size=32)
python logreg_train.py datasets/dataset_train.csv -m mini-batch
3. Inference
Apply your trained model weights to categorize new students:

python logreg_predict.py datasets/dataset_test.csv
This generates houses.csv with the Sorting Hat's final assignments.

4. Running Benchmark Harnesses
To automatically run, evaluate, and benchmark all three methods side-by-side:

python evaluate.py datasets/dataset_train.csv
📊 Comparative Benchmarks
Evaluation metrics computed directly across all gradient descent variations on a 1200-student dataset:

Optimization Method	Epochs Run	Learning Rate ($\alpha$)	Convergence Speed	Training Accuracy	Meets 98% Threshold?
Batch Gradient Descent	150	0.50	5.7281s	100.00%	Yes ✅
Stochastic Gradient Descent	15	0.01	0.6475s	100.00%	Yes ✅
Mini-Batch Gradient Descent	30	0.10	1.3335s	100.00%	Yes ✅
🛡️ License
This project is licensed under the MIT License - see the LICENSE file for details.

Created with mathematical precision by an elite Machine Learning Engineer.
