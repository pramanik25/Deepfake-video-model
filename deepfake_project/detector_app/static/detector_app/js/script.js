const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("videoUpload");
const successMsg = document.getElementById("successMsg");
const resultArea = document.getElementById("resultArea"); 


dropArea.addEventListener("click", () => fileInput.click());

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.style.background = "rgba(78, 204, 163, 0.15)";
});

dropArea.addEventListener("dragleave", () => {
  dropArea.style.background = "rgba(255, 255, 255, 0.05)";
});

dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            // Optionally, display the filename
            dropArea.querySelector('p').textContent = `File: ${e.dataTransfer.files[0].name}`;
        }
        dropArea.style.background = "rgba(255, 255, 255, 0.05)";
    });

    
// Update dropArea text if a file is selected via click
if (fileInput) {
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            dropArea.querySelector('p').textContent = `File: ${fileInput.files[0].name}`;
        } else {
            dropArea.querySelector('p').textContent = 'Click to Upload or Drag & Drop';
        }
    });
}

function uploadVideo() {
  // const isLoggedIn = localStorage.getItem("loggedIn") === "true";

  //if (!isLoggedIn) {
   // alert("Please log in to upload a video.");
   // window.location.href = "login.html"; // redirect to your login page
   // return;
 // }

  const fileInput = document.getElementById("videoUpload");
  const successMsg = document.getElementById("successMsg");


  if (fileInput.files.length === 0) {
    alert("Please select a video file first.");
    return;
  }

  
    successMsg.textContent = "⏳ Uploading and processing video...";
    successMsg.style.display = "block";
    resultArea.innerHTML = ""; // Clear previous results

    const formData = new FormData();
    formData.append('videoFile', fileInput.files[0]); // 'videoFile' must match Django view

    // Get CSRF token from the page
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    fetch('/predict/', { // The URL of your Django endpoint
        method: 'POST',
        headers: {
            'X-CSRFToken': csrfToken // Include CSRF token
        },
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            // Try to get error message from Django if it's JSON
            return response.json().then(err => { throw new Error(err.error || `Server error: ${response.status}`) });
        }
        return response.json();
    })
    .then(data => {
        console.log("Success:", data);
        successMsg.textContent = "✅ Processing complete!";
        if (data.error) {
            resultArea.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            let probabilityPercent = (data.probability * 100).toFixed(2);
            resultArea.innerHTML = `
                <p>Prediction: <strong style="color: ${data.prediction === 'Fake' ? 'orange' : 'lightgreen'};">${data.prediction}</strong></p>
                <p>Confidence: ${probabilityPercent}%</p>
            `;
        }
        // Hide success message after a delay, or keep it
        setTimeout(() => {
            // successMsg.style.display = "none"; // Or update it
        }, 5000);
    })
    .catch((error) => {
        console.error("Error:", error);
        successMsg.textContent = "❌ Error during processing.";
        resultArea.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
        // setTimeout(() => {
        //     successMsg.style.display = "none";
        // }, 5000);
    })
    .finally(() => {
        // Reset file input and drop area text
        fileInput.value = ""; // Clear the selected file
        if (dropArea) {
           dropArea.querySelector('p').textContent = 'Click to Upload or Drag & Drop';
        }
    });
}
  


// Optional: Add smooth scrolling with more control
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
  
      const targetID = document.querySelector(this.getAttribute('href'));
  
      targetID.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    });
  });

  



  