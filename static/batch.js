// static/batch.js
// ================= BATCH RESUME SCREENING (ES MODULE) =================

export function initBatchScreening() {
    const batchForm = document.querySelector('form[action="/batch-screen"]');
    if (!batchForm) return;

    batchForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(batchForm);

        try {
            const res = await fetch("/batch-screen", {
                method: "POST",
                body: formData
            });

            const contentType = res.headers.get("content-type") || "";

            // ✅ FALLBACK: Server returned HTML → render it
            if (!contentType.includes("application/json")) {
                const html = await res.text();
                document.open();
                document.write(html);
                document.close();
                return;
            }

            const data = await res.json();

            if (!data.ranked_results || !Array.isArray(data.ranked_results)) {
                throw new Error("Invalid batch response format");
            }

            // ---- RENDER RESULTS (SINGLE-STYLE UI) ----
            const container = document.getElementById("dynamic-results");
            if (!container) return;

            container.innerHTML = "";

            data.ranked_results.forEach((r, index) => {
                const status = r.final >= 65 ? "PASS" : "FAIL";
                const risk = r.final < 60 ? "HIGH" : "LOW";

                container.innerHTML += `
                <section class="results glass-card result-landscape">
                    <div class="result-col status-col">
                        <div class="status-pill ${status === "PASS" ? "pass" : "fail"}">
                            ${status}
                        </div>
                        <div class="status-card">
                            <h3>#${index + 1} ${r.filename}</h3>
                            <p>ATS Confidence: 70%</p>
                        </div>
                    </div>

                    <div class="result-col metrics-col">
                        <div class="metric-card">
                            <span class="metric-label">JD Coverage</span>
                            <span class="metric-value blue">${r.coverage}%</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">Hiring Risk</span>
                            <span class="metric-value ${risk === "HIGH" ? "danger" : "success"}">
                                ${risk}
                            </span>
                        </div>
                    </div>

                    <div class="result-col score-col">
                        <div class="final-score">
                            <span>Final Score</span>
                            <h2>${r.final.toFixed(1)}%</h2>
                        </div>
                        <div class="mini-scores">
                            <div class="mini-card">
                                Quality<br><strong>${r.quality}%</strong>
                            </div>
                            <div class="mini-card">
                                Job Match<br><strong>${r.match}%</strong>
                            </div>
                        </div>
                    </div>
                </section>
                `;
            });

        } catch (err) {
            console.error("Batch screening failed:", err);
        }
    });
}

/* ======================================================
   ✅ FIXED: Batch Compare Checkbox Logic (ES MODULE SAFE)
====================================================== */

export function initBatchCompare() {
    const form = document.getElementById("batch-compare-form");
    if (!form) return;

    const compareBtn = document.getElementById("compare-btn");
    const checkboxes = form.querySelectorAll(".compare-checkbox");

    if (compareBtn) {
        compareBtn.disabled = true;
    }

    checkboxes.forEach(cb => {
        cb.addEventListener("change", () => {
            const selected = [...checkboxes].filter(c => c.checked);

            // Max 2 selections
            if (selected.length > 2) {
                cb.checked = false;
                return;
            }

            // Enable only when exactly 2 selected
            if (compareBtn) {
                compareBtn.disabled = selected.length !== 2;
            }
        });
    });
}
