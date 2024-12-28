from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt

# 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터셋 나누기 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 결정 트리 모델 생성
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 모델 평가
accuracy = model.score(X_test, y_test)
print(f"모델 정확도: {accuracy:.2f}")

# 결정 트리 시각화
plt.figure(figsize=(10,8))
tree.plot_tree(model, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.show()

# 텍스트 형태로 출력
tree_rules = export_text(model, feature_names=wine.feature_names)
print(tree_rules)