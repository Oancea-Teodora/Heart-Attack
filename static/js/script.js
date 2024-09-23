    document.getElementById('prediction-form').addEventListener('submit', function(e) {
        e.preventDefault();  // Prevent the form from submitting the usual way

        // Show the progress bar
        var progressBar = document.getElementById('progress-bar');
        var progressBarContainer = document.getElementById('progress-bar-container');
        progressBarContainer.style.display = 'block'; // Show the progress bar container

        // Reset the progress bar
        progressBar.style.width = '0%';

        var progress = 0;
        var progressInterval = setInterval(function() {
            if (progress < 90) {  // Increment until 90% to simulate waiting
                progress += 10;
                progressBar.style.width = progress + '%';
            } else {
                clearInterval(progressInterval);
            }
        }, 300);  // Increment the progress every 300 milliseconds (0.3 seconds)

        var formData = new FormData(this);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Clear the progress bar once the prediction result is received
            clearInterval(progressInterval);

            // Simulate a 100% progress completion
            progressBar.style.width = '100%';

            // Hide the progress bar after a short delay
            setTimeout(function() {
                progressBarContainer.style.display = 'none';
                progressBar.style.width = '0%';
            }, 500);

            // Display the prediction result
            document.getElementById('result').innerText = data.prediction;
        })
        .catch(error => {
            console.error('Error:', error);

            // Hide the progress bar in case of an error
            clearInterval(progressInterval);
            progressBarContainer.style.display = 'none';
            progressBar.style.width = '0%';

            document.getElementById('result').innerText = 'An error occurred: ' + error.message;
        });
    });