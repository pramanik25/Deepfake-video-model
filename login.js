// Toggle light/dark mode
const toggleBtn = document.getElementById('toggleMode');
toggleBtn.addEventListener('click', () => {
  document.body.classList.toggle('light-mode');
});



// Show spinner on login
const loginForm = document.getElementById('loginForm');
const spinner = document.getElementById('spinner');
const loginBtn = document.getElementById('loginBtn');

loginForm.addEventListener('submit', (e) => {
  e.preventDefault();
  spinner.style.display = "block";
  loginBtn.disabled = true;
  loginBtn.innerText = "Logging in...";

  setTimeout(() => {
    spinner.style.display = "none";
    loginBtn.disabled = false;
    loginBtn.innerText = "Login";
    alert('Logged in successfully!');
  }, 3000); // fake 3 second login
});

// Simulate successful login
document.getElementById("loginForm").addEventListener("submit", function (e) {
  e.preventDefault();

  // Assume login is successful
  localStorage.setItem("loggedIn", "true");

  // Redirect to home or upload page
  window.location.href = "index.html"; // adjust path as needed
});
