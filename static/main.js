// static/main.js
import { initBatchScreening } from "./batch.js";
import { initResumeComparison } from "./compare.js";

document.addEventListener("DOMContentLoaded", () => {
    initBatchScreening();
    initResumeComparison();
});
