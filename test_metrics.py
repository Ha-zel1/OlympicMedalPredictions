from model_enhanced import train_and_predict_model

results = train_and_predict_model()

print("\n" + "="*50)
print("MODEL METRICS (Should match notebook)")
print("="*50)

print("\nLinear Regression:")
print(f"  R²: {results['all_metrics']['linear_regression']['r2_score']:.6f}")
print(f"  RMSE: {results['all_metrics']['linear_regression']['rmse']:.6f}")
print(f"  Expected: R² = 0.760285, RMSE = 8.989165")

print("\nGradient Boosting:")
print(f"  R²: {results['all_metrics']['gradient_boosting']['r2_score']:.6f}")
print(f"  RMSE: {results['all_metrics']['gradient_boosting']['rmse']:.6f}")
print(f"  Expected: R² = 0.917673, RMSE = 5.267972")

print("\nSVM:")
print(f"  Accuracy: {results['all_metrics']['svm']['accuracy']:.6f}")
print(f"  Expected: Accuracy = 0.610000")

print("\n" + "="*50)
