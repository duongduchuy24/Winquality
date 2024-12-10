import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# 1. Load dataset
data = pd.read_csv('D:\\Nam_3\\CSTTNT\\prj2_decision_tree\\wine+quality\\winequality-red.csv', sep=';')
X = data.drop('quality', axis=1)
y = data['quality']

# 2. Define splits
proportions = [(0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)]
splits = []

for train_size, test_size in proportions:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=42)
    splits.append((X_train, X_test, y_train, y_test))

# Định nghĩa độ sâu tối đa của cây
max_depth_value = 5  # Bạn có thể thay đổi giá trị này để điều chỉnh độ sâu của cây


# 3. Visualize distributions

# Original dataset distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=y)
plt.title("Original Dataset Distribution")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.show(block=False)

for i, (X_train, X_test, y_train, y_test) in enumerate(splits):

    # Train set distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=y_train)
    plt.title(f"Train Set Distribution ({proportions[i][0] * 100}% Train)")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    plt.show(block=False)

    # Test set distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=y_test)
    plt.title(f"Test Set Distribution ({proportions[i][1] * 100}% Test)")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    plt.show(block=False)

# Ensure all figures are displayed after the code finishes
plt.show()


for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth_value, random_state=42)
    clf.fit(X_train, y_train)
    
    # Xuất cây quyết định dưới dạng DOT format
    dot_data = export_graphviz(clf, out_file=None, 
                               feature_names=X.columns,  
                               class_names=[str(c) for c in clf.classes_],  
                               filled=True, 
                               rounded=True,  
                               special_characters=True)  
    
    # Tạo và lưu cây quyết định
    graph = graphviz.Source(dot_data)
    graph.render(f"decision_tree_split_{i + 1}_train", format='png', view=True)  # Lưu và mở cây dưới dạng PNG

    #print(f"Decision Tree for Split {i + 1} ({proportions[i][0] * 100}% Train): Rendered")


    # Dự đoán trên tập kiểm tra
for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth_value, random_state=42)
    clf.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = clf.predict(X_test)
    
    # In classification report
    print(f"\nClassification Report for Split {i + 1} ({proportions[i][1] * 100}% Test):")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # In confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f"Confusion Matrix for Split {i + 1}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show(block = False)
plt.show()

#5
# Thử các giá trị khác nhau cho tham số max_depth
depth_values = [None, 2, 3, 4, 5, 6, 7]
accuracy_results = []

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y, random_state=42)

for depth in depth_values:
    # Khởi tạo và huấn luyện mô hình với max_depth khác nhau
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = clf.predict(X_test)
    
    # Tính accuracy và lưu kết quả
    acc = accuracy_score(y_test, y_pred)
    accuracy_results.append((depth, acc))
    
    # In báo cáo và confusion matrix
    print(f"\nMax Depth = {depth}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

# Vẽ biểu đồ ảnh hưởng của max_depth đến accuracy
depth_labels = ['None' if d is None else str(d) for d in depth_values]
accuracies = [acc for _, acc in accuracy_results]

plt.figure(figsize=(8, 6))
plt.plot(depth_labels, accuracies, marker='o', linestyle='-', color='b')
plt.title("Effect of Tree Depth on Classification Accuracy (80/20 Split)")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()