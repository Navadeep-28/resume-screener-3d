// ================= IMPORT THREE.JS =================
import * as THREE from 'three';

// ================= DYNAMIC RESUME SCREENER =================

document.addEventListener("DOMContentLoaded", () => {
    init3DCanvas();
    initForm();
    initSpinnerDots();
    initJobTemplates();
});

/* ================= JOB DESCRIPTION TEMPLATES ================= */
const JOB_TEMPLATES = {
    python_dev: `
Looking for a Python Developer with experience in Flask, Django,
REST APIs, SQL databases, and basic machine learning concepts.
Strong problem-solving skills required.
`,

    data_scientist: `
Seeking a Data Scientist skilled in Python, Pandas, NumPy,
Machine Learning, Statistics, and Data Visualization.
NLP experience is a plus.
`,

    ml_engineer: `
Machine Learning Engineer with hands-on experience in
model training, deployment, scikit-learn, TensorFlow or PyTorch,
and NLP pipelines.
`,

    frontend_dev: `
Frontend Developer with strong knowledge of HTML, CSS, JavaScript,
React, responsive design, and UI/UX principles.
`,

    backend_dev: `
Backend Developer experienced in APIs, databases, authentication,
Python or Java, and system design concepts.
`
};

/* ================= TEMPLATE INIT ================= */
function initJobTemplates() {
    const roleSelect = document.getElementById("job-role");
    const jdTextarea = document.getElementById("job-desc");

    if (!roleSelect || !jdTextarea) return;

    roleSelect.addEventListener("change", () => {
        const role = roleSelect.value;
        if (JOB_TEMPLATES[role]) {
            jdTextarea.value = JOB_TEMPLATES[role].trim();
        }
    });
}

/* ================= 3D BACKGROUND ================= */
function init3DCanvas() {
    const canvas = document.getElementById("three-canvas");
    if (!canvas) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );

    const renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);

    const particles = new THREE.BufferGeometry();
    const count = window.innerWidth < 768 ? 400 : 1000;
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < positions.length; i += 3) {
        positions[i] = (Math.random() - 0.5) * 20;
        positions[i + 1] = (Math.random() - 0.5) * 20;
        positions[i + 2] = (Math.random() - 0.5) * 20;
    }

    particles.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        color: 0x00e6ff,
        size: 0.05,  // Increased from 0.008 - try values between 0.01 and 0.1
        transparent: true,
        opacity: 0.8
        sizeAttenuation: true
    });

    const particleSystem = new THREE.Points(particles, material);
    scene.add(particleSystem);

    camera.position.z = 5;

    function animate() {
        requestAnimationFrame(animate);
        particleSystem.rotation.y += 0.0005;
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

/* ================= FORM SUBMISSION ================= */
function initForm() {
    const form = document.getElementById("upload-form");
    const spinner = document.getElementById("spinner");
    const container = document.querySelector(".container");

    if (!form || !spinner || !container) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        spinner.style.display = "block";
        form.style.opacity = "0.6";

        try {
            const response = await fetch("/", {
                method: "POST",
                body: new FormData(form)
            });

            if (!response.ok) throw new Error("Server error");

            const html = await response.text();
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, "text/html");

            const newResults = doc.querySelector(".results");
            const newError = doc.querySelector(".error");

            container.querySelectorAll(".results, .error").forEach(el => el.remove());

            if (newError) container.appendChild(newError);

            if (newResults) {
                container.appendChild(newResults);

                setTimeout(() => {
                    renderSkillsChart();
                    applyHeatmap();
                }, 100);
            }

        } catch (err) {
            console.error(err);
            container.insertAdjacentHTML(
                "beforeend",
                '<div class="error">❌ Network or server error. Check terminal.</div>'
            );
        } finally {
            spinner.style.display = "none";
            form.style.opacity = "1";
        }
    });
}

/* ================= SKILLS CHART ================= */
function renderSkillsChart() {
    if (!window.Plotly) return;

    const skills = [];
    document.querySelectorAll(".skills-list li").forEach(li => {
        const match = li.textContent.match(/(.+?) \((\d+)%\)/);
        if (match) {
            skills.push({
                name: match[1],
                score: Number(match[2]) / 100
            });
        }
    });

    if (!skills.length) return;

    Plotly.newPlot("skills-chart", [{
        values: skills.slice(0, 6).map(s => s.score),
        labels: skills.slice(0, 6).map(s => s.name),
        type: "pie",
        hole: 0.4,
        textinfo: "label+percent",
        textposition: "outside",
        marker: {
            colors: ["#00e6ff", "#667eea", "#764ba2", "#ff6b9d", "#00ff88", "#ffd23f"]
        }
    }], {
        title: { text: "💎 Top Skills Match", font: { size: 18 } },
        showlegend: false,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)"
    }, { responsive: true });
}

/* ================= WORD HEATMAP ================= */
function applyHeatmap() {
    const preview = document.getElementById("resume-preview");
    if (!preview || !preview.dataset.heatmap) return;

    const heatWords = JSON.parse(preview.dataset.heatmap);
    let html = preview.innerText;

    heatWords.forEach(word => {
        const regex = new RegExp(`\\b(${word})\\b`, "gi");
        html = html.replace(regex, `<span class="heat-word">$1</span>`);
    });

    preview.innerHTML = html;
}

/* ================= SPINNER DOTS ================= */
function initSpinnerDots() {
    const dots = document.querySelector(".dots");
    if (!dots) return;

    let step = 0;
    setInterval(() => {
        step = (step + 1) % 4;
        dots.textContent = ".".repeat(step);
    }, 300);
}
