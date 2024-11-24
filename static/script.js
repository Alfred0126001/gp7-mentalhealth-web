document.getElementById("simulation-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const scenario = document.getElementById("scenario").value;

    const response = await fetch("/run_simulation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario }),
    });

    const data = await response.json();
    document.getElementById("results").textContent = JSON.stringify(data.results, null, 2);
});