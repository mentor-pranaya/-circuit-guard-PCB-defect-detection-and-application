document.addEventListener('DOMContentLoaded', () => {
    const refInput = document.getElementById('ref-input');
    const testInput = document.getElementById('test-input');
    const refPreview = document.getElementById('ref-preview');
    const testPreview = document.getElementById('test-preview');
    const detectBtn = document.getElementById('detect-btn');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');
    const annotatedImage = document.getElementById('annotated-image');
    const predictionsList = document.getElementById('predictions-list');

    // Function to handle image preview
    const setupPreview = (input, preview) => {
        input.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    };

    setupPreview(refInput, refPreview);
    setupPreview(testInput, testPreview);

    // Main detection logic
    detectBtn.addEventListener('click', async () => {
        const refFile = refInput.files[0];
        const testFile = testInput.files[0];

        if (!refFile || !testFile) {
            alert('Please upload both a reference and a test image.');
            return;
        }

        // --- UI updates for processing ---
        detectBtn.disabled = true;
        detectBtn.textContent = 'Processing...';
        resultsSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        annotatedImage.style.display = 'none';
        predictionsList.innerHTML = '';

        // --- Prepare data and call API ---
        const formData = new FormData();
        formData.append('ref_image', refFile);
        formData.append('test_image', testFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'An unknown error occurred.');
            }
            
            // --- Display results ---
            displayResults(result);

        } catch (error) {
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('hidden');
            console.error('Prediction failed:', error);
        } finally {
            // --- Reset UI ---
            loader.classList.add('hidden');
            detectBtn.disabled = false;
            detectBtn.textContent = '🚀 Detect Defects';
        }
    });

    function displayResults(data) {
        // Display annotated image
        annotatedImage.src = data.annotated_image;
        annotatedImage.style.display = 'block';

        // Display predictions list
        if (data.predictions && data.predictions.length > 0) {
            data.predictions.forEach(pred => {
                const li = document.createElement('li');
                li.textContent = `Defect: ${pred.label} (Confidence: ${pred.confidence})`;
                predictionsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No defects found.';
            predictionsList.appendChild(li);
        }
    }
});
