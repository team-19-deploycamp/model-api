const apiUrl = 'http://localhost:8000';  // Ganti jika endpoint FastAPI berbeda

document.getElementById('load-places-btn').addEventListener('click', async () => {
  const res = await fetch(`${apiUrl}/cold_start`);
  const places = await res.json();

  const list = document.getElementById('places-list');
  list.innerHTML = '';

  places.forEach(place => {
    const div = document.createElement('div');
    div.innerHTML = `
      <p><strong>${place.Place_Name}</strong> - ${place.Category} (${place.City})</p>
      <label>Rating: <input type="number" min="1" max="5" step="1" data-place-id="${place.Place_Id}"></label>
      <hr/>
    `;
    list.appendChild(div);
  });
});

document.getElementById('preference-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const userId = parseInt(document.getElementById('user-id').value);
  const inputs = document.querySelectorAll('[data-place-id]');
  
  const preferences = Array.from(inputs)
    .filter(input => input.value)  // hanya yang diisi rating
    .map(input => ({
      Place_Id: parseInt(input.dataset.placeId),
      Rating: parseFloat(input.value)
    }));

  const payload = {
    User_Id: userId,
    Preferences: preferences
  };

  const res = await fetch(`${apiUrl}/submit_preference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await res.json();

  const list = document.getElementById('recommendations-list');
  list.innerHTML = '';

  data.Recommendations.forEach(rec => {
    const li = document.createElement('li');
    li.textContent = `${rec.Place_Name} (${rec.Category}, ${rec.City}) - Score: ${rec.Score}`;
    list.appendChild(li);
  });
});
