async function loadTableData() {
    console.log("JavaScript werkt! loadTableData() wordt aangeroepen.");
    const response = await fetch("/api/labeling_tasks");
    console.log("Fetched response:", response);
    const result = await response.json();
    console.log("Parsed JSON result:", result);
    const data = result; 

    const tbody = document.querySelector("tbody");
    console.log("Selected tbody:", tbody);
    tbody.innerHTML = "";

    data.sort((a, b) => new Date(b.received) - new Date(a.received));
    console.log("Sorted data:", data);

    for (const row of data) {
        console.log("Processing row:", row);
        const tr = document.createElement("tr");

        const userTd = `<td data-label="User">${row.user}</td>`;
        const totalTd = `<td data-label="Total Images">${row.total_images}</td>`;
        const progressTd = `
            <td data-label="Labeled %">
                <div class="progress">
                    <div class="progress-bar" style="width: ${row.labeled_percentage}%;">${row.labeled_percentage}%</div>
                </div>
            </td>`;
        const timestampTd = `<td data-label="Received">${row.received}</td>`;

        tr.innerHTML = userTd + totalTd + progressTd + timestampTd;

        tr.style.cursor = "pointer";
        tr.addEventListener("click", () => {
            console.log(`Row clicked, navigating to /label?dataset=${encodeURIComponent(row.dataset_id)}`);
            window.location.href = `/label?dataset=${encodeURIComponent(row.dataset_id)}`;
        });

        tbody.appendChild(tr);
    }
    console.log("Table data loaded.");
}

document.addEventListener("DOMContentLoaded", loadTableData);
