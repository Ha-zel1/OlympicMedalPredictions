document.getElementById("predictBtn").addEventListener("click", function() {
  fetch('/predict', { method: 'POST' })
  .then(response => response.json())
  .then(data => {
      if (data.success) {
          document.getElementById("comparisonImg").src = "/static/images/comparison.png";
      }
  })
  .catch(error => console.error('Error:', error));
});
