// static/mode-switch.js
export function initModeSwitch() {
    const buttons = document.querySelectorAll(".mode-btn");
    const form = document.querySelector("form");
    if (!form) return;

    const singleInput = form.querySelector('input[name="resume"]');
    const batchBlock = form.querySelector(".mode-batch");
    const compareBlock = form.querySelector(".mode-compare");

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const mode = btn.dataset.mode;

            // Reset visibility
            batchBlock.style.display = "none";
            compareBlock.style.display = "none";
            singleInput.style.display = "block";

            // Reset form action
            form.action = "/";

            if (mode === "batch") {
                singleInput.style.display = "none";
                batchBlock.style.display = "block";
                form.action = "/batch-screen";
            }

            if (mode === "compare") {
                compareBlock.style.display = "block";
                form.action = "/compare-resumes";
            }
        });
    });
}
