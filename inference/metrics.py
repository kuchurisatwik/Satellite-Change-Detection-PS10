# Evaluate the model
test_loss, test_accuracy = model_unet.evaluate(X_test, y_test, batch_size=8, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predict on test set
predictions_unet = model_unet.predict(X_test, batch_size=8, verbose=1)
#predicted_masks = (predictions > 0.5).astype(int)