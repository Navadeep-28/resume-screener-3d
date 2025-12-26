// ========== 3D BACKGROUND SETUP ==========
let scene, camera, renderer, stars, ringMesh, clock;

function initThreeJS() {
    const canvas = document.getElementById('three-canvas');

    if (!canvas || typeof THREE === 'undefined') {
        console.warn('Three.js or canvas not available, skipping 3D background');
        return;
    }

    scene = new THREE.Scene();
    clock = new THREE.Clock();

    camera = new THREE.PerspectiveCamera(
        70,
        window.innerWidth / window.innerHeight,
        0.1,
        1500
    );
    camera.position.set(0, 0, 140);

    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    canvas.appendChild(renderer.domElement);

    // Neon ambient + point light
    const ambient = new THREE.AmbientLight(0x442266, 0.8);
    scene.add(ambient);

    const pointLight = new THREE.PointLight(0x00e6ff, 1.2, 600);
    pointLight.position.set(0, 80, 120);
    scene.add(pointLight);

    // Particle star field
    const particles = 800;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particles * 3);

    for (let i = 0; i < positions.length; i += 3) {
        positions[i]     = (Math.random() - 0.5) * 2000;
        positions[i + 1] = (Math.random() - 0.5) * 1200;
        positions[i + 2] = (Math.random() - 0.5) * 2000;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
        color: 0x00e6ff,
        size: 3,
        transparent: true,
        opacity: 0.8
    });
    stars = new THREE.Points(geometry, material);
    scene.add(stars);

    // Neon torus ring
    const ringGeom = new THREE.TorusBufferGeometry(50, 4, 32, 120);
    const ringMat = new THREE.MeshStandardMaterial({
        color: 0xff00ff,
        emissive: 0x550055,
        metalness: 0.7,
        roughness: 0.25
    });
    ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringMesh.rotation.x = Math.PI / 3;
    scene.add(ringMesh);

    animate();
}

function animate() {
    if (!renderer || !scene || !camera) return;

    requestAnimationFrame(animate);

    const t = clock ? clock.getElapsedTime() : performance.now() * 0.001;

    if (stars) {
        stars.rotation.x += 0.00025;
        stars.rotation.y += 0.0004;
    }

    if (ringMesh) {
        ringMesh.rotation.y += 0.004;
        ringMesh.rotation.z = Math.sin(t * 0.6) * 0.4;
    }

    // Subtle camera orbit
    const radius = 140;
    camera.position.x = Math.cos(t * 0.1) * radius;
    camera.position.z = Math.sin(t * 0.1) * radius;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
}

// Handle resize safely
window.addEventListener('resize', () => {
    if (!camera || !renderer) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ========== FORM & ANALYSIS LOGIC ==========
document.addEventListener('DOMContentLoaded', function () {
    // Start 3D background
    initThreeJS();

    const form    = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    if (!form) return;

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        if (loading) loading.classList.remove('d-none');
        if (results) results.classList.add('d-none');

        const formData = new FormData(form);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data.results, data.pdf_url);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            alert('Error analyzing resume: ' + error.message);
        } finally {
            if (loading) loading.classList.add('d-none');
        }
    });
});

function displayResults(results, pdfUrl) {
    const resultsDiv    = document.getElementById('results');
    const scoreCircle   = document.getElementById('overallScore');
    const downloadBtn   = document.getElementById('downloadBtn');
    const jobTitle      = document.getElementById('jobTitle');
    const decisionText  = document.getElementById('decisionText');
    const skillsSummary = document.getElementById('skillsSummary');
        const blindUsed = results.blind_mode;

    if (blindUsed && decisionText) {
        decisionText.textContent += ' (Blind screening: identity fields hidden from the model.)';
    }


    if (resultsDiv) resultsDiv.classList.remove('d-none');

    // Overall score circle
    if (scoreCircle) {
        const overall = Number(results.overall || 0);
        scoreCircle.textContent = overall.toFixed(1) + '%';
        const accepted = overall >= 70;
        scoreCircle.style.background = accepted
            ? 'radial-gradient(circle, #3cff9c 0%, #00ffd0 40%, #006f52 100%)'
            : 'radial-gradient(circle, #ff6b6b 0%, #ff2e63 40%, #7f0000 100%)';
        if (decisionText) {
            decisionText.textContent = accepted
                ? 'This resume meets the threshold for this role.'
                : 'This resume currently scores below the threshold for this role.';
        }
    }

    if (jobTitle && results.job_category) {
        jobTitle.textContent = results.job_category;
    }

    if (downloadBtn) {
        downloadBtn.href = pdfUrl;
    }

    if (skillsSummary) {
        const matched = (results.matched_skills || []).join(', ') || 'None';
        const missing = (results.missing_skills || []).join(', ') || 'None';
        skillsSummary.innerHTML =
            `<b>Matched skills:</b> ${matched}<br>` +
            `<b>Missing skills:</b> ${missing}`;
    }

    // Explainability panel
    const explainTop   = document.getElementById('explainTopPhrases');
    const explainMiss  = document.getElementById('explainMissingCritical');
    const explainExp   = document.getElementById('explainExperience');

    const explanation = results.explanation || {};

    if (explainTop) {
        explainTop.innerHTML = '';
        (explanation.top_matching_phrases || []).forEach(p => {
            const li = document.createElement('li');
            li.textContent = p;
            explainTop.appendChild(li);
        });
        if (!explanation.top_matching_phrases || !explanation.top_matching_phrases.length) {
            explainTop.innerHTML = '<li>No strong matches detected.</li>';
        }
    }

    if (explainMiss) {
        explainMiss.innerHTML = '';
        (explanation.missing_critical_skills || []).forEach(s => {
            const li = document.createElement('li');
            li.textContent = s;
            explainMiss.appendChild(li);
        });
        if (!explanation.missing_critical_skills || !explanation.missing_critical_skills.length) {
            explainMiss.innerHTML = '<li>No critical gaps found.</li>';
        }
    }

    if (explainExp) {
        explainExp.textContent = explanation.experience_explanation ||
            'Experience score is based on extracted years and seniority terms in the resume.';
    }

    createScoreCharts(results);
}

// ========== PLOTLY CHARTS WITH NEON THEME ==========
function neonLayout(title) {
    // Shared dark-neon layout
    return {
        title: { text: title, font: { color: '#ffffff', size: 16 } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e5e5ff' },
        margin: { l: 40, r: 20, t: 40, b: 40 }
    };
}

function createScoreCharts(results) {
    const pieContainer  = document.getElementById('scorePie');
    const barContainer  = document.getElementById('scoreBars');
    const heatContainer = document.getElementById('overlapHeatmap');

    if (typeof Plotly === 'undefined') {
        console.warn('Plotly not available – charts will not render');
        return;
    }

    const tfidfScore = Number(results.tfidf || 0);
    const skillScore = Number(results.skills || 0);
    const expScore   = Number(results.experience || 0);

    // PIE: score components (neon donut)
    if (pieContainer) {
        const pieData = [{
            type: 'pie',
            values: [tfidfScore, skillScore, expScore],
            labels: ['TF‑IDF match', 'Skills match', 'Experience'],
            hole: 0.45,
            marker: {
                colors: ['#00e6ff', '#ff00ff', '#feca57'],
                line: { color: '#05010a', width: 2 }
            },
            textinfo: 'label+percent',
            hoverinfo: 'label+value+percent',
            pull: [0.02, 0.04, 0.02]
        }];

        const layout = neonLayout('Score components');
        Plotly.newPlot(pieContainer, pieData, layout, { displayModeBar: false });
    }

    // BAR: exact percentage scores
    if (barContainer) {
        const barData = [{
            x: ['TF‑IDF', 'Skills', 'Experience'],
            y: [tfidfScore, skillScore, expScore],
            type: 'bar',
            marker: {
                color: ['#00e6ff', '#ff00ff', '#3cff9c'],
                line: { color: '#ffffff', width: 1 }
            },
            text: [tfidfScore, skillScore, expScore].map(v => v.toFixed(1) + '%'),
            textposition: 'outside',
            hoverinfo: 'x+y'
        }];

        const layout = neonLayout('Scores (%)');
        layout.yaxis = { range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)' };
        layout.xaxis = { gridcolor: 'rgba(255,255,255,0.05)' };

        Plotly.newPlot(barContainer, barData, layout, { displayModeBar: false });
    }

    // HEATMAP: overlap statistics
    if (heatContainer && results.heatmap_data) {
        const h = results.heatmap_data;
        const common       = Number(h.common_words || 0);
        const resumeUnique = Number(h.resume_unique || 0);
        const jdUnique     = Number(h.jd_unique || 0);
        const commonPct    = Number(h.common_percentage || 0);

        const heatData = [{
            z: [[common, resumeUnique, jdUnique]],
            x: ['Common', 'Resume‑only', 'JD‑only'],
            y: ['Overlap'],
            type: 'heatmap',
            colorscale: [
                [0.0, '#1b0034'],
                [0.3, '#00e6ff'],
                [0.6, '#ff00ff'],
                [1.0, '#ff6b6b']
            ],
            showscale: true,
            hoverongaps: false
        }];

        const layout = neonLayout(`Word overlap (common: ${commonPct.toFixed(1)}%)`);
        layout.yaxis = { showgrid: false };
        layout.xaxis = { showgrid: false };

        Plotly.newPlot(heatContainer, heatData, layout, { displayModeBar: false });
    }
}
