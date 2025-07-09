function parseCustomTimestamp(ts) {
    const parts = ts.split("-");
    if (parts.length < 6) return new Date(""); 

    const [year, month, day, hour, minute, second] = parts;
    return new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`);
}

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

    data.sort((a, b) => parseCustomTimestamp(b.timestamp) - parseCustomTimestamp(a.timestamp));
    console.log("Sorted data:", data);

    for (const row of data) {
        console.log("Processing row:", row);
        const tr = document.createElement("tr");

        const userTd = `<td data-label="User">${row.user}</td>`;
        const totalTd = `<td data-label="Total Images">${row.total_images}</td>`;
        const labeledPercentage = Math.round((row.num_labeled / row.total_images) * 100);

        const progressTd = `
            <td data-label="Labeled %">
                <div class="progress">
                    <div class="progress-bar" style="width: ${labeledPercentage}%;">${labeledPercentage}%</div>
                </div>
            </td>`;

        const date = parseCustomTimestamp(row.timestamp);
        const formattedDate = isNaN(date)
            ? "Onbekend"
            : date.toLocaleString("nl-NL", {
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit"
            });
        const timestampTd = `<td data-label="Received">${formattedDate}</td>`;
        console.log("Raw timestamp:", row.timestamp, "| Parsed:", date);

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