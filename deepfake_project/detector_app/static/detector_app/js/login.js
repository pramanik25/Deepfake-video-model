// particlesJS config from your script.js should be here if this page uses it independently
// Or ensure script.js (which has particlesJS) is loaded on login.html too if particles are desired there.
// For now, login.html directly loads the CDN and calls particlesJS with its own config.

const loginForm = document.getElementById('loginForm'); // Make sure your form has id="loginForm"
const spinner = document.getElementById('spinner');
const loginBtn = document.getElementById('loginBtn'); // Make sure your button has id="loginBtn"
const toggleBtn = document.getElementById('toggleMode'); // Ensure this button exists in login.html

if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      document.body.classList.toggle('light-mode');
    });
}

if (loginForm && loginBtn) { // Check if elements exist
    loginForm.addEventListener('submit', (e) => {
        // Spinner logic can remain if you want visual feedback before Django redirects
        if (spinner) spinner.style.display = "block";
        if (loginBtn) {
            loginBtn.disabled = true;
            loginBtn.innerText = "Logging in...";
        }
        // No e.preventDefault(); here, let the form submit to Django
        // No setTimeout or manual redirection; Django will handle it.
        // No localStorage.setItem("loggedIn", "true"); - Django handles auth
    });
}

// If using particles.js on this page, and you have a particles-config.js:
// Assuming particles-config.js defines a global variable like `particlesConfiguration`
// particlesJS('particles-js', particlesConfiguration); 
// OR, if login.html has its own inline config or CDN + separate call as you've shown, that's fine.
// Your login.html currently has:
// <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
// <script src="particles-config.js"></script> <!-- THIS WILL BE A 404 unless you mean {% static '... %} -->
// It should be:
// <script src="{% static 'detector_app/js/particles-config.js' %}"></script> 
// OR include the config directly in login.js

// Example of particle config (if not in a separate file linked correctly)
// This should match the one in your script.js if you want consistency
if (typeof particlesJS !== 'undefined') { // Check if particlesJS is loaded
    particlesJS("particles-js", {
        "particles": {
          "number": {"value": 80, "density": {"enable": true, "value_area": 700}},
          "color": {"value": "#ffffff"},
          "shape": {"type": "circle"},
          "opacity": {"value": 0.5, "random": true},
          "size": {"value": 4, "random": true},
          "line_linked": {"enable": true, "distance": 150, "color": "#ffffff", "opacity": 0.4, "width": 1},
          "move": {"enable": true, "speed": 3, "out_mode": "out"}
        },
        "interactivity": {
          "events": {"onhover": {"enable": true, "mode": "grab"}, "onclick": {"enable": true, "mode": "push"}},
          "modes": {"grab": {"distance": 200, "line_linked": {"opacity": 0.5}}, "push": {"particles_nb": 4}}
        },
        "retina_detect": true
      });
} else {
    console.warn("particles.js not loaded on login page.");
}