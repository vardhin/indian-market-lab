async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {}
    throw new Error(message);
  }

  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  health: () => request('/api/health'),
  experiments: () => request('/api/experiments'),
  scripts: () => request('/api/scripts'),
  runs: () => request('/api/runs'),
  run: (id) => request(`/api/runs/${id}`),
  logs: (id, after = 0) => request(`/api/runs/${id}/logs?after=${after}`),
  startRun: (experiment_id, config, name = null) =>
    request('/api/runs', { method: 'POST', body: JSON.stringify({ experiment_id, config, name }) }),
  startCustom: (script, args = [], name = null) =>
    request('/api/runs/custom', { method: 'POST', body: JSON.stringify({ script, args, name }) }),
  stopRun: (id) => request(`/api/runs/${id}/stop`, { method: 'POST' }),
  marketYears: () => request('/api/market/years'),
  symbols: (year, query = '') => request(`/api/market/symbols?year=${year}&query=${encodeURIComponent(query)}`),
  candles: (symbol, year, interval, endDate = null) => {
    const end = endDate ? `&end_date=${encodeURIComponent(endDate)}` : '';
    return request(`/api/market/candles?symbol=${encodeURIComponent(symbol)}&year=${year}&interval=${interval}${end}`);
  },
  dashboard: (year = 2023) => request(`/api/dashboard?year=${year}`),
  reports: () => request('/api/reports'),
  artifact: (path, limit = 1000) => request(`/api/artifact?path=${encodeURIComponent(path)}&limit=${limit}`),
  createGame: (config) => request('/api/game', { method: 'POST', body: JSON.stringify(config) }),
  game: (id) => request(`/api/game/${id}`),
  gameAction: (id, action) => request(`/api/game/${id}/action`, { method: 'POST', body: JSON.stringify(action) }),
  gameMarket: (id, query = '') => request(`/api/game/${id}/market?query=${encodeURIComponent(query)}`)
};
