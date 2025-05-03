const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("videoUpload");
const successMsg = document.getElementById("successMsg");

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
  fileInput.files = e.dataTransfer.files;
  dropArea.style.background = "rgba(255, 255, 255, 0.05)";
});

function uploadVideo() {
  const isLoggedIn = localStorage.getItem("loggedIn") === "true";

  if (!isLoggedIn) {
    alert("Please log in to upload a video.");
    window.location.href = "login.html"; // redirect to your login page
    return;
  }

  const fileInput = document.getElementById("videoUpload");
  const successMsg = document.getElementById("successMsg");


  if (fileInput.files.length === 0) {
    alert("Please select a video file first.");
    return;
  }

  successMsg.style.display = "block";
  setTimeout(() => {
    successMsg.style.display = "none";
    fileInput.value = "";
  }, 3000);
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

  



  