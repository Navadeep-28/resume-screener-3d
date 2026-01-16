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

            // ---- SAFETY CHECK: JSON vs HTML ----
            const contentType = res.headers.get("content-type") || "";

            if (!contentType.includes("application/json")) {
                const text = await res.text();
                console.error("Expected JSON but received:", text);
                throw new Error("Server returned HTML instead of JSON");
            }

            const data = await res.json();

            if (!data.ranked_results || !Array.isArray(data.ranked_results)) {
                throw new Error("Invalid batch response format");
            }

            // ---- RENDER RESULTS ----
            let output = `<h3>📊 Ranked Resumes</h3><ol>`;
            data.ranked_results.forEach(r => {
                output += `
                    <li>
                        <strong>${r.filename}</strong><br>
                        Final Score: ${r.final.toFixed(1)}%<br>
                        JD Coverage: ${r.coverage}%
                    </li>
                `;
            });
            output += `</ol>`;

            let resultBox = document.getElementById("batch_results");
            if (!resultBox) {
                resultBox = document.createElement("div");
                resultBox.id = "batch_results";
                resultBox.className = "glass-card";
                batchForm.parentElement.appendChild(resultBox);
            }

            resultBox.innerHTML = output;

        } catch (err) {
            console.error("Batch screening failed:", err);

            let resultBox = document.getElementById("batch_results");
            if (!resultBox) {
                resultBox = document.createElement("div");
                resultBox.id = "batch_results";
                resultBox.className = "glass-card error";
                batchForm.parentElement.appendChild(resultBox);
            }

            resultBox.innerHTML = `
                <p>❌ Batch screening failed.</p>
                <small>Check console or backend logs.</small>
            `;
        }
    });
}
