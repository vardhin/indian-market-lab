<script>
  import { onMount } from 'svelte';
  import MarketChart from '$lib/MarketChart.svelte';
  import MetricCard from '$lib/MetricCard.svelte';
  import { api } from '$lib/api.js';

  let mode = 'research';
  let dockTab = 'runs';
  let health = null;
  let experiments = [];
  let scripts = [];
  let customScript = '';
  let customArgs = '';
  let runs = [];
  let dashboard = { metrics: {}, decisions: [], equity: [], reports: [] };
  let actionMatrix = { state_date: null, rows: [] };
  let reports = [];
  let artifactView = null;
  let artifactLoading = false;
  let years = [2023];

  let experimentId = 'portfolio_oracle_v4';
  let config = {};
  let selectedRunId = null;
  let logLines = [];
  let logCursor = 0;
  let runPoll;

  let year = 2023;
  let symbol = 'RELIANCE';
  let symbolQuery = 'RELIANCE';
  let symbols = [];
  let candles = [];
  let interval = '1D';
  let chartType = 'candles';
  let upColor = '#35d07f';
  let downColor = '#ff5d6c';
  let gridColor = '#1c2330';
  let background = '#0b0f15';
  let showVolume = true;
  let showGrid = true;
  let candleStatus = 'idle';
  let ohlcQuality = 'unknown';
  let timelineIndex = 0;

  let game = null;
  let gameMarket = [];
  let gameYear = 2023;
  let gameStartDate = '';
  let gameHistoryBars = 260;
  let gameCapital = 50000;
  let gameMaxHoldings = 5;
  let gameSearch = '';
  let gameSelected = 'RELIANCE';
  let switchFrom = '';

  let orderSizingMode = 'shares';
  let orderQuantity = 1;
  let orderAmount = 10000;
  let orderFractionPct = 10;
  let orderTargetWeightPct = 20;
  let holdingModalSymbol = '';

  let toast = null;
  let toastTimer;

  $: selectedExperiment = experiments.find((item) => item.id === experimentId);
  $: selectedRun = runs.find((item) => item.id === selectedRunId);
  $: metrics = mode === 'game' && game ? game.metrics || {} : dashboard.metrics || {};
  $: decisions = mode === 'game' && game ? game.history || [] : dashboard.decisions || [];
  $: currentDecision = decisions.length
    ? decisions[Math.min(timelineIndex, decisions.length - 1)]
    : null;
  $: markers = buildMarkers();
  $: selectedPosition =
    game?.holdings?.find(
      (holding) =>
        holding.symbol === gameSelected
    ) || null;
  $: selectedMarketRow =
    gameMarket?.find(
      (item) =>
        item.symbol === gameSelected
    ) || null;
  $: modalPosition =
    game?.holdings?.find(
      (holding) =>
        holding.symbol === holdingModalSymbol
    ) || null;

  function notify(message, type = 'info') {
    toast = { message, type };
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), 3600);
  }

  function fmtMoney(value) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(Number(value));
  }

  function fmtPct(value, digits = 2) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function fmtNum(value, digits = 3) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(digits);
  }

  function defaultsFor(experiment) {
    const next = {};
    for (const p of experiment?.params || []) {
      next[p.name] = Array.isArray(p.default)
        ? p.default.join(', ')
        : p.default ?? '';
    }
    return next;
  }

  function chooseExperiment(id) {
    experimentId = id;
    if (id === '__custom__') {
      if (!customScript && scripts.length) {
        customScript = scripts[0];
      }
      config = {};
      return;
    }
    const exp = experiments.find((item) => item.id === id);
    config = defaultsFor(exp);
  }

  function parseCustomArgs(text) {
    return (text.match(/(?:[^\\s"]+|"[^"]*")+/g) || [])
      .map((part) => part.replace(/^"|"$/g, ''));
  }

  async function setMode(nextMode) {
    mode = nextMode;
    await loadCandles();
  }

  function normalizeConfig() {
    const out = {};
    for (const p of selectedExperiment?.params || []) {
      let value = config[p.name];

      if (p.type === 'int_list' || p.type === 'str_list') {
        value = String(value ?? '')
          .split(',')
          .map((v) => v.trim())
          .filter(Boolean);

        if (p.type === 'int_list') {
          value = value.map(Number);
        }
      } else if (p.type === 'int' || p.type === 'optional_int') {
        value = String(value ?? '').trim() === ''
          ? null
          : Number.parseInt(value, 10);
      } else if (p.type === 'float') {
        value = String(value ?? '').trim() === ''
          ? null
          : Number(value);
      }

      out[p.name] = value;
    }
    return out;
  }

  async function launchExperiment() {
    try {
      const run = experimentId === '__custom__'
        ? await api.startCustom(
            customScript,
            parseCustomArgs(customArgs)
          )
        : await api.startRun(
            experimentId,
            normalizeConfig()
          );
      selectedRunId = run.id;
      dockTab = 'logs';
      logLines = [];
      logCursor = 0;
      await refreshRuns();
      notify(`Started ${run.name}`);
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  async function refreshRuns() {
    try {
      runs = await api.runs();
      if (!selectedRunId && runs.length) {
        selectedRunId = runs[0].id;
      }
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  async function refreshLogs() {
    if (!selectedRunId) return;
    try {
      const payload = await api.logs(
        selectedRunId,
        logCursor
      );
      if (payload.lines.length) {
        logLines = [
          ...logLines,
          ...payload.lines
        ];
        logCursor = payload.next;
      }
    } catch {}
  }

  async function stopRun() {
    if (!selectedRunId) return;
    try {
      await api.stopRun(
        selectedRunId
      );
      await refreshRuns();
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  async function loadActionMatrix(date = null) {
    if (mode === 'game') return;
    try {
      actionMatrix = await api.actionMatrix(
        date
      );
    } catch (error) {
      actionMatrix = {
        state_date: date,
        rows: []
      };
      console.error(error);
    }
  }

  async function refreshDashboard() {
    try {
      const [nextDashboard, nextReports] =
        await Promise.all([
          api.dashboard(year),
          api.reports()
        ]);
      dashboard = nextDashboard;
      reports = nextReports;
      if (dashboard.decisions?.length) {
        timelineIndex =
          dashboard.decisions.length - 1;
      }
      await loadActionMatrix(
        dashboard.decisions?.length
          ? String(
              dashboard.decisions[
                dashboard.decisions.length - 1
              ].date || ''
            ).slice(0, 10)
          : null
      );
    } catch (error) {
      console.error(error);
    }
  }

  async function loadArtifact(path) {
    artifactLoading = true;
    try {
      artifactView = await api.artifact(
        path,
        500
      );
    } catch (error) {
      notify(error.message, 'error');
    } finally {
      artifactLoading = false;
    }
  }

  async function searchSymbols() {
    try {
      symbols = await api.symbols(
        year,
        symbolQuery
      );
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  async function loadCandles() {
    if (!symbol) return;

    candleStatus = 'loading';
    try {
      const payload = await api.candles(
        symbol,
        year,
        interval,
        mode === 'game' && game
          ? game.date
          : null,
        mode === 'game' && game
          ? game.history_bars
          : null
      );
      candles = payload.candles;
      ohlcQuality = payload.ohlc_quality || 'unknown';
      if (
        ohlcQuality !== 'true_ohlc'
        && chartType === 'candles'
      ) {
        chartType = 'line';
      }
      candleStatus = `${candles.length} bars`;
    } catch (error) {
      candleStatus = 'error';
      notify(error.message, 'error');
    }
  }

  function selectSymbol(next) {
    symbol = next;
    symbolQuery = next;
    symbols = [];
    loadCandles();
  }

  function actionTargetLabel(row) {
    if (!row?.target?.length) {
      return '100% CASH';
    }
    return row.target
      .map(
        (item) =>
          `${item.asset} ${(
            Number(item.weight) * 100
          ).toFixed(1)}%`
      )
      .join(' · ');
  }

  function parseTargetAssets(row) {
    if (!row) return '';

    if (row.target_assets) {
      try {
        return JSON.parse(
          row.target_assets
        ).join(', ');
      } catch {
        return row.target_assets;
      }
    }

    if (row.to_symbol) {
      return `${row.from_symbol || 'cash'} → ${row.to_symbol}`;
    }

    return row.symbol || row.action || '—';
  }

  function buildMarkers() {
    const source =
      mode === 'game' && game
        ? game.history || []
        : dashboard.decisions || [];

    return source
      .slice(-80)
      .map((row) => {
        const time = (
          row.date
          || row.signal_date
          || ''
        ).slice(0, 10);

        if (!time) return null;

        const action =
          row.action
          || (row.target_assets
            ? 'AGENT'
            : 'ACTION');

        const sellish =
          action === 'SELL'
          || action === 'LIQUIDATE';

        return {
          time,
          position: sellish
            ? 'aboveBar'
            : 'belowBar',
          color: sellish
            ? '#ff5d6c'
            : '#7c95ff',
          shape: sellish
            ? 'arrowDown'
            : 'circle',
          text: action
        };
      })
      .filter(Boolean);
  }

  async function startGame() {
    try {
      game = await api.createGame({
        year: Number(gameYear),
        start_date:
          String(gameStartDate || '').trim()
            || null,
        history_bars:
          Number(gameHistoryBars),
        initial_capital:
          Number(gameCapital),
        max_holdings:
          Number(gameMaxHoldings)
      });

      mode = 'game';
      year = game.year;
      gameStartDate =
        game.start_date || '';
      timelineIndex = 0;
      await refreshGameMarket();
      await loadCandles();

      notify(
        'Game started. Use the game date as your decision boundary; actions execute at the next market open.'
      );
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  async function refreshGameMarket() {
    if (!game?.id) return;

    try {
      gameMarket = await api.gameMarket(
        game.id,
        gameSearch
      );

      if (
        !gameSelected
        && gameMarket.length
      ) {
        gameSelected =
          gameMarket[0].symbol;
      }
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  async function act(
    type,
    extra = {}
  ) {
    if (!game?.id) return false;

    try {
      game = await api.gameAction(
        game.id,
        {
          type,
          ...extra
        }
      );
      timelineIndex = Math.max(
        0,
        (game.history?.length || 1) - 1
      );
      await refreshGameMarket();
      await loadCandles();

      if (type === 'LIQUIDATE') {
        notify(
          `Liquidated at ${fmtMoney(game.equity)}`
        );
      }
      return true;
    } catch (error) {
      notify(error.message, 'error');
      return false;
    }
  }

  function orderPayload() {
    if (orderSizingMode === 'shares') {
      return {
        sizing_mode: 'shares',
        quantity: Math.max(
          1,
          Number.parseInt(
            orderQuantity,
            10
          ) || 1
        )
      };
    }

    if (orderSizingMode === 'rupees') {
      return {
        sizing_mode: 'rupees',
        amount: Math.max(
          1,
          Number(orderAmount) || 1
        )
      };
    }

    if (orderSizingMode === 'equity_fraction') {
      return {
        sizing_mode: 'equity_fraction',
        fraction: Math.min(
          1,
          Math.max(
            0.0001,
            Number(orderFractionPct)
            / 100
          )
        )
      };
    }

    return {
      sizing_mode: 'target_weight',
      target_weight: Math.min(
        1,
        Math.max(
          0,
          Number(orderTargetWeightPct)
          / 100
        )
      )
    };
  }

  function buyOrAdd() {
    if (!gameSelected) return;
    act(
      'BUY',
      {
        symbol: gameSelected,
        ...orderPayload()
      }
    );
  }

  function trimOrSell() {
    if (!gameSelected) return;
    act(
      'SELL',
      {
        symbol: gameSelected,
        ...orderPayload()
      }
    );
  }

  function setTargetWeight() {
    if (!gameSelected) return;
    act(
      'SET_TARGET',
      {
        symbol: gameSelected,
        sizing_mode:
          'target_weight',
        target_weight:
          Math.min(
            1,
            Math.max(
              0,
              Number(
                orderTargetWeightPct
              )
              / 100
            )
          )
      }
    );
  }

  function openHoldingModal(holding) {
    holdingModalSymbol =
      holding.symbol;
    gameSelected =
      holding.symbol;
    orderTargetWeightPct =
      Math.max(
        0,
        Math.min(
          100,
          Number(
            holding.weight
          )
          * 100
        )
      );
    orderQuantity = 1;
    selectSymbol(
      holding.symbol
    );
  }

  function closeHoldingModal() {
    holdingModalSymbol = '';
  }

  async function modalAdd() {
    if (!modalPosition) return;
    const ok = await act(
      'BUY',
      {
        symbol:
          modalPosition.symbol,
        ...orderPayload()
      }
    );
    if (ok) closeHoldingModal();
  }

  async function modalTrim() {
    if (!modalPosition) return;
    const ok = await act(
      'SELL',
      {
        symbol:
          modalPosition.symbol,
        ...orderPayload()
      }
    );
    if (ok) closeHoldingModal();
  }

  async function modalSetTarget() {
    if (!modalPosition) return;
    const ok = await act(
      'SET_TARGET',
      {
        symbol:
          modalPosition.symbol,
        sizing_mode:
          'target_weight',
        target_weight:
          Math.min(
            1,
            Math.max(
              0,
              Number(
                orderTargetWeightPct
              )
              / 100
            )
          )
      }
    );
    if (ok) closeHoldingModal();
  }

  async function modalExitPosition() {
    if (!modalPosition) return;
    const ok = await act(
      'SELL',
      {
        symbol:
          modalPosition.symbol,
        sizing_mode:
          'shares',
        quantity:
          modalPosition.quantity
      }
    );
    if (ok) closeHoldingModal();
  }

  function switchAction() {
    if (!switchFrom || !gameSelected) {
      return;
    }

    act(
      'SWITCH',
      {
        from_symbol: switchFrom,
        to_symbol: gameSelected
      }
    );
  }

  onMount(async () => {
    try {
      [health, experiments, scripts, years] =
        await Promise.all([
          api.health(),
          api.experiments(),
          api.scripts(),
          api.marketYears()
        ]);

      if (years.includes(2023)) {
        year = 2023;
      } else if (years.length) {
        year =
          years[years.length - 1];
      }

      gameYear = year;
      chooseExperiment(
        experimentId
      );

      await Promise.all([
        refreshRuns(),
        refreshDashboard(),
        searchSymbols()
      ]);

      if (
        symbols.length
        && !symbols.some(
          (item) =>
            item.symbol === symbol
        )
      ) {
        symbol = symbols[0].symbol;
      }

      await loadCandles();
    } catch (error) {
      notify(error.message, 'error');
    }

    runPoll = setInterval(
      async () => {
        await refreshRuns();
        await refreshLogs();
      },
      1200
    );

    return () => {
      clearInterval(
        runPoll
      );
    };
  });

  $: if (selectedRunId) {
    const index = runs.findIndex(
      (run) =>
        run.id === selectedRunId
    );

    if (
      index >= 0
      && runs[index]?.status
        === 'completed'
    ) {
      refreshDashboard();
    }
  }
</script>

<svelte:head>
  <title>Indian Market Lab · Agent Console</title>
  <meta
    name="description"
    content="Research, backtesting, portfolio-agent and market-game console"
  />
</svelte:head>

<div class="lab-shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark"></div>
      <div>
        <div class="brand-title">
          Indian Market Lab
        </div>
        <div class="brand-sub">
          portfolio agent research console
        </div>
      </div>
    </div>

    <div class="mode-tabs">
      <button
        class:active={mode === 'research'}
        class="mode-tab"
        onclick={() =>
          setMode('research')}
      >
        Research
      </button>
      <button
        class:active={mode === 'game'}
        class="mode-tab"
        onclick={() =>
          setMode('game')}
      >
        Human Game
      </button>
    </div>

    <div class="top-spacer"></div>
    <div class="health">
      <span
        class:ok={health?.ok}
        class="health-dot"
      ></span>
      {health?.ok
        ? `API · Python ${health.python}`
        : 'API offline'}
    </div>
  </header>

  <main class="workspace">
    <aside class="panel sidebar">
      <div class="panel-header">
        <div class="panel-title">
          {mode === 'research'
            ? 'Experiment'
            : 'Game setup'}
        </div>
        <div class="panel-subtitle">
          control
        </div>
      </div>

      <div class="panel-body">
        {#if mode === 'research'}
          <div class="section">
            <div class="control">
              <label>Experiment</label>
              <select
                class="select"
                bind:value={experimentId}
                onchange={(event) =>
                  chooseExperiment(
                    event.currentTarget.value
                  )}
              >
                {#each experiments as exp}
                  <option value={exp.id}>
                    {exp.group} · {exp.label}
                  </option>
                {/each}
                <option value="__custom__">
                  Advanced · any src/ml script
                </option>
              </select>
            </div>
            <div class="tiny muted">
              {experimentId === '__custom__'
                ? 'Launch any Python experiment under src/ml without a shell.'
                : selectedExperiment?.description || ''}
            </div>
          </div>

          <div class="section">
            <div class="section-title">
              Configuration
            </div>

            {#if experimentId === '__custom__'}
              <div class="control">
                <label>Python script</label>
                <select
                  class="select"
                  bind:value={customScript}
                >
                  {#each scripts as script}
                    <option value={script}>
                      {script}
                    </option>
                  {/each}
                </select>
              </div>
              <div class="control">
                <label>CLI arguments</label>
                <textarea
                  class="textarea mono"
                  bind:value={customArgs}
                  placeholder='--year 2023 --capital 50000'
                ></textarea>
              </div>
            {:else}
              {#each selectedExperiment?.params || [] as p}
                <div class="control">
                  <label>
                    {p.label || p.name}
                  </label>

                  {#if p.choices && p.type !== 'str_list'}
                    <select
                      class="select"
                      bind:value={config[p.name]}
                    >
                      {#each p.choices as choice}
                        <option value={choice}>
                          {choice}
                        </option>
                      {/each}
                    </select>
                  {:else}
                    <input
                      class="input"
                      type={
                        p.type === 'int'
                        || p.type === 'float'
                        || p.type === 'optional_int'
                          ? 'number'
                          : 'text'
                      }
                      min={p.min ?? undefined}
                      max={p.max ?? undefined}
                      step={p.type === 'float'
                        ? 'any'
                        : undefined}
                      bind:value={config[p.name]}
                      placeholder={p.type.endsWith('_list')
                        ? 'comma separated'
                        : ''}
                    />
                  {/if}
                </div>
              {/each}
            {/if}

            <button
              class="btn btn-primary"
              style="width:100%"
              onclick={launchExperiment}
            >
              {experimentId === '__custom__'
                ? 'Run script'
                : 'Run experiment'}
            </button>
          </div>

          <div class="section">
            <div class="section-title">
              Active run
            </div>

            {#if selectedRun}
              <div
                class="tiny mono"
                style="overflow-wrap:anywhere"
              >
                {selectedRun.name}
              </div>

              <div style="margin-top:7px">
                <span
                  class={`status ${selectedRun.status}`}
                >
                  {selectedRun.status}
                </span>
              </div>

              <div
                class="btn-row"
                style="margin-top:9px"
              >
                <button
                  class="btn"
                  onclick={() =>
                    (dockTab = 'logs')}
                >
                  Logs
                </button>

                {#if selectedRun.status === 'running' || selectedRun.status === 'queued'}
                  <button
                    class="btn btn-danger"
                    onclick={stopRun}
                  >
                    Stop
                  </button>
                {/if}
              </div>
            {:else}
              <div class="tiny muted">
                No run selected.
              </div>
            {/if}
          </div>
        {:else}
          <div class="section">
            <div class="inline-controls">
              <div class="control">
                <label>Year</label>
                <select
                  class="select"
                  bind:value={gameYear}
                >
                  {#each years as y}
                    <option value={y}>
                      {y}
                    </option>
                  {/each}
                </select>
              </div>

              <div class="control">
                <label>
                  Max holdings
                </label>
                <input
                  class="input"
                  type="number"
                  min="1"
                  max="20"
                  bind:value={gameMaxHoldings}
                />
              </div>
            </div>

            <div class="inline-controls">
              <div class="control">
                <label>
                  Start date
                </label>
                <input
                  class="input"
                  type="date"
                  bind:value={gameStartDate}
                />
              </div>

              <div class="control">
                <label>
                  Prior history bars
                </label>
                <input
                  class="input"
                  type="number"
                  min="20"
                  max="2000"
                  step="20"
                  bind:value={gameHistoryBars}
                />
              </div>
            </div>

            <div class="control">
              <label>
                Initial capital
              </label>
              <input
                class="input"
                type="number"
                min="1000"
                step="1000"
                bind:value={gameCapital}
              />
            </div>

            <button
              class="btn btn-primary"
              style="width:100%"
              onclick={startGame}
            >
              {game
                ? 'Restart game'
                : 'Start game'}
            </button>
          </div>

          {#if game}
            <div class="section">
              <div
                class="section-title"
                style="display:flex;justify-content:space-between;gap:8px"
              >
                <span>
                  Market at {game.date}
                </span>
                <span
                  class="muted"
                  style="font-weight:500;letter-spacing:0;text-transform:none"
                >
                  {gameMarket.length} eligible stocks
                </span>
              </div>

              <div class="control">
                <label>
                  Search current universe
                </label>
                <input
                  class="input"
                  bind:value={gameSearch}
                  oninput={refreshGameMarket}
                  placeholder="TCS, INFY…"
                />
              </div>

              <div class="game-market">
                {#each gameMarket as item}
                  <button
                    class:active={gameSelected === item.symbol}
                    class="game-symbol"
                    onclick={() => {
                      gameSelected =
                        item.symbol;
                      selectSymbol(
                        item.symbol
                      );
                    }}
                  >
                    <strong>
                      {item.symbol}
                    </strong>
                    <span>
                      {item.close?.toFixed?.(2) ?? '—'}
                    </span>
                  </button>
                {/each}
              </div>
            </div>

            <div class="section">
              <div class="section-title">
                Your action
              </div>

              <div
                class="tiny muted"
                style="margin-bottom:8px"
              >
                Selected:
                <strong
                  style="color:#dce5f0"
                >
                  {gameSelected || '—'}
                </strong>
                {#if selectedPosition}
                  · already holding
                  <strong
                    style="color:#76e6aa"
                  >
                    {selectedPosition.quantity}
                    shares
                  </strong>
                {/if}
              </div>

              <div
                class="tiny mono"
                style="
                  padding:7px 8px;
                  border:1px solid var(--line);
                  border-radius:6px;
                  background:#0a0e14;
                  margin-bottom:9px;
                  line-height:1.6;
                "
              >
                Minimum order: 1 whole share · no fractional shares
                {#if selectedMarketRow?.close}
                  <br />
                  Close: {fmtMoney(
                    selectedMarketRow.close
                  )}
                  · approx max buy from cash:
                  {Math.max(
                    0,
                    Math.floor(
                      game.cash
                      / selectedMarketRow.close
                    )
                  )} shares
                {/if}
              </div>

              <div class="control">
                <label>
                  Sizing mode
                </label>
                <select
                  class="select"
                  bind:value={orderSizingMode}
                >
                  <option value="shares">
                    Shares
                  </option>
                  <option value="rupees">
                    ₹ amount
                  </option>
                  <option value="equity_fraction">
                    % of portfolio equity
                  </option>
                  <option value="target_weight">
                    Target portfolio weight
                  </option>
                </select>
              </div>

              {#if orderSizingMode === 'shares'}
                <div class="control">
                  <label>
                    Shares
                  </label>
                  <input
                    class="input"
                    type="number"
                    min="1"
                    step="1"
                    bind:value={orderQuantity}
                  />
                </div>
              {:else if orderSizingMode === 'rupees'}
                <div class="control">
                  <label>
                    Rupee notional
                  </label>
                  <input
                    class="input"
                    type="number"
                    min="1"
                    step="1000"
                    bind:value={orderAmount}
                  />
                </div>
              {:else if orderSizingMode === 'equity_fraction'}
                <div class="control">
                  <label>
                    Equity fraction (%)
                  </label>
                  <input
                    class="input"
                    type="number"
                    min="0.01"
                    max="100"
                    step="1"
                    bind:value={orderFractionPct}
                  />
                </div>
              {:else}
                <div class="control">
                  <label>
                    Target weight (%)
                  </label>
                  <input
                    class="input"
                    type="number"
                    min="0"
                    max="100"
                    step="1"
                    bind:value={orderTargetWeightPct}
                  />
                </div>
              {/if}

              <div class="game-actions">
                {#if orderSizingMode === 'target_weight'}
                  <button
                    class="btn btn-good"
                    onclick={setTargetWeight}
                  >
                    Set target
                  </button>
                {:else}
                  <button
                    class="btn btn-good"
                    onclick={buyOrAdd}
                  >
                    {selectedPosition
                      ? 'Add / average'
                      : 'Buy'}
                  </button>

                  <button
                    class="btn btn-danger"
                    onclick={trimOrSell}
                  >
                    Trim / sell
                  </button>
                {/if}

                <button
                  class="btn"
                  onclick={() =>
                    act('HOLD')}
                >
                  Hold / next
                </button>

                <button
                  class="btn btn-danger"
                  onclick={() =>
                    act(
                      'LIQUIDATE'
                    )}
                >
                  Liquidate all
                </button>
              </div>

              <div
                class="control"
                style="margin-top:9px"
              >
                <label>
                  Switch from holding → selected
                </label>
                <select
                  class="select"
                  bind:value={switchFrom}
                >
                  <option value="">
                    Choose holding
                  </option>
                  {#each game.holdings || [] as holding}
                    <option
                      value={holding.symbol}
                    >
                      {holding.symbol}
                    </option>
                  {/each}
                </select>
              </div>

              <button
                class="btn"
                style="width:100%"
                onclick={switchAction}
              >
                Switch → {gameSelected || '…'}
              </button>
            </div>
          {/if}
        {/if}
      </div>
    </aside>

    <section class="panel chart-panel">
      <div class="panel-header">
        <div class="panel-title">
          Market
        </div>
        <div class="panel-subtitle">
          price / actions
        </div>
      </div>

      <div class="chart-toolbar">
        <input
          class="input symbol-input"
          bind:value={symbolQuery}
          onkeydown={(event) =>
            event.key === 'Enter'
            && selectSymbol(
              symbolQuery.toUpperCase()
            )}
        />

        <button
          class="btn"
          onclick={() =>
            selectSymbol(
              symbolQuery.toUpperCase()
            )}
        >
          Load
        </button>

        <select
          class="select toolbar-select"
          bind:value={year}
          onchange={async () => {
            await searchSymbols();
            await loadCandles();
            await refreshDashboard();
          }}
        >
          {#each years as y}
            <option value={y}>
              {y}
            </option>
          {/each}
        </select>

        <select
          class="select toolbar-select"
          bind:value={interval}
          onchange={loadCandles}
        >
          <option>1D</option>
          <option>1W</option>
          <option>1M</option>
          <option>3M</option>
        </select>

        <select
          class="select toolbar-select"
          bind:value={chartType}
        >
          <option
            value="candles"
            disabled={ohlcQuality !== 'true_ohlc'}
          >
            Candles
          </option>
          <option value="line">
            Line
          </option>
          <option value="area">
            Area
          </option>
        </select>

        <div class="toolbar-group">
          <span class="toolbar-label">
            Up
          </span>
          <input
            class="color-input"
            type="color"
            bind:value={upColor}
          />
          <span class="toolbar-label">
            Down
          </span>
          <input
            class="color-input"
            type="color"
            bind:value={downColor}
          />
          <span class="toolbar-label">
            Grid
          </span>
          <input
            class="color-input"
            type="color"
            bind:value={gridColor}
          />
        </div>

        <div class="toolbar-group">
          <label class="tiny muted">
            <input
              type="checkbox"
              bind:checked={showVolume}
            />
            Vol
          </label>
          <label class="tiny muted">
            <input
              type="checkbox"
              bind:checked={showGrid}
            />
            Grid
          </label>
        </div>

        <div class="chart-status">
          {symbol} · {candleStatus}
        </div>
      </div>

      {#if symbols.length > 0 && symbolQuery && symbolQuery !== symbol}
        <div
          style="
            position:absolute;
            z-index:10;
            margin:4px 0 0 10px;
            width:210px;
            background:#0d1219;
            border:1px solid #2a3545;
            border-radius:6px;
            max-height:180px;
            overflow:auto;
          "
        >
          {#each symbols.slice(0, 12) as item}
            <button
              class="game-symbol"
              onclick={() =>
                selectSymbol(
                  item.symbol
                )}
            >
              {item.symbol}
            </button>
          {/each}
        </div>
      {/if}

      <MarketChart
        {candles}
        {chartType}
        {upColor}
        {downColor}
        {gridColor}
        {background}
        {showVolume}
        {showGrid}
        {markers}
        height={520}
      />

      <div class="chart-footer">
        <span>
          <strong>{interval}</strong>
          research bars
        </span>
        <span>
          OHLC · volume · action markers
        </span>

        {#if mode === 'game' && game}
          <span>
            decision boundary:
            <strong>{game.date}</strong>
          </span>
        {/if}

        <span style="margin-left:auto">
          {ohlcQuality === 'true_ohlc'
            ? 'True NSE OHLCV; longer intervals are server-side OHLC resamples.'
            : 'Open/close-only fallback: line/area mode only.'}
        </span>
      </div>
    </section>

    <aside class="panel telemetry">
      <div class="panel-header">
        <div class="panel-title">
          {mode === 'game'
            ? 'Player state'
            : 'Agent state'}
        </div>
        <div class="panel-subtitle">
          telemetry
        </div>
      </div>

      <div class="panel-body">
        <div class="metric-grid">
          <MetricCard
            label="Capital"
            value={
              mode === 'game'
              && game
                ? fmtMoney(
                    game.equity
                  )
                : fmtMoney(
                    metrics.ending_equity
                    || metrics.ending_capital
                  )
            }
          />

          <MetricCard
            label="Fees"
            value={
              mode === 'game'
              && game
                ? fmtMoney(
                    game.fees
                  )
                : fmtMoney(
                    metrics.total_fees
                    || metrics.fees
                  )
            }
          />

          {#if mode === 'game' && game}
            <MetricCard
              label="Realized P&L"
              value={fmtMoney(
                game.realized_pnl
              )}
              tone={
                game.realized_pnl > 0
                  ? 'good'
                  : game.realized_pnl < 0
                    ? 'bad'
                    : 'neutral'
              }
            />
          {/if}

          <MetricCard
            label="CAGR"
            value={fmtPct(
              metrics.cagr
            )}
            tone={
              metrics.cagr > 0
                ? 'good'
                : metrics.cagr < 0
                  ? 'bad'
                  : 'neutral'
            }
          />

          <MetricCard
            label="Sharpe"
            value={fmtNum(
              metrics.sharpe
            )}
            tone={
              metrics.sharpe > 1
                ? 'good'
                : metrics.sharpe < 0
                  ? 'bad'
                  : 'neutral'
            }
          />

          <MetricCard
            label="Sortino"
            value={fmtNum(
              metrics.sortino
            )}
          />

          <MetricCard
            label="Max DD"
            value={fmtPct(
              metrics.max_drawdown
            )}
            tone="bad"
          />

          <MetricCard
            label="Calmar"
            value={fmtNum(
              metrics.calmar
            )}
          />

          <MetricCard
            label="Volatility"
            value={fmtPct(
              metrics.annualized_volatility
            )}
          />

          <MetricCard
            label="Total return"
            value={fmtPct(
              metrics.total_return
            )}
            tone={
              metrics.total_return > 0
                ? 'good'
                : metrics.total_return < 0
                  ? 'bad'
                  : 'neutral'
            }
          />

          <MetricCard
            label="Turnover"
            value={
              metrics.turnover_multiple == null
                ? '—'
                : `${fmtNum(
                    metrics.turnover_multiple,
                    1
                  )}×`
            }
          />

          <MetricCard
            label="Exposure"
            value={
              mode === 'game'
              && game
                ? fmtPct(
                    1
                    - game.cash
                    / Math.max(
                      game.equity,
                      1
                    )
                  )
                : fmtPct(
                    metrics.average_exposure
                  )
            }
          />

          <MetricCard
            label="Holdings"
            value={
              mode === 'game'
              && game
                ? String(
                    game.holdings
                      ?.length
                    || 0
                  )
                : fmtNum(
                    metrics.average_holdings,
                    2
                  )
            }
          />
        </div>

        <div class="section">
          <div class="section-title">
            {mode === 'game'
              ? 'Holdings'
              : 'Recent actions'}
          </div>

          {#if mode === 'game' && game}
            {#if game.holdings?.length}
              <div class="position-card-grid">
                {#each game.holdings as holding}
                  <button
                    class:profit={holding.unrealized_pnl > 0}
                    class:loss={holding.unrealized_pnl < 0}
                    class:flat={holding.unrealized_pnl === 0}
                    class="position-card"
                    onclick={() =>
                      openHoldingModal(
                        holding
                      )}
                  >
                    <div class="position-card-head">
                      <div>
                        <div class="position-symbol">
                          {holding.symbol}
                        </div>
                        <div class="position-weight">
                          {fmtPct(
                            holding.weight
                          )}
                          of portfolio
                        </div>
                      </div>

                      <div class="position-return">
                        {holding.unrealized_pnl >= 0
                          ? '+'
                          : ''}{fmtPct(
                          holding.unrealized_return
                        )}
                      </div>
                    </div>

                    <div class="position-value">
                      {fmtMoney(
                        holding.value
                      )}
                    </div>

                    <div class="position-card-stats">
                      <span>
                        <strong>{holding.quantity}</strong>
                        shares
                      </span>
                      <span>
                        avg
                        <strong>{fmtMoney(
                          holding.average_cost
                        )}</strong>
                      </span>
                      <span>
                        now
                        <strong>{fmtMoney(
                          holding.price
                        )}</strong>
                      </span>
                    </div>

                    <div class="position-pnl-row">
                      <span>
                        unrealized
                      </span>
                      <strong>
                        {holding.unrealized_pnl >= 0
                          ? '+'
                          : ''}{fmtMoney(
                          holding.unrealized_pnl
                        )}
                      </strong>
                    </div>

                    <div class="position-card-foot">
                      Click to manage position
                    </div>
                  </button>
                {/each}
              </div>
            {:else}
              <div class="tiny muted">
                100% cash.
              </div>
            {/if}
          {:else}
            <div class="action-list">
              {#each (dashboard.decisions || []).slice(-8).reverse() as row}
                <div class="action-row">
                  <div class="action-date">
                    {String(
                      row.date || ''
                    ).slice(5, 10)}
                  </div>
                  <div>
                    {parseTargetAssets(row)}
                  </div>
                  <div class="action-q">
                    Q {fmtNum(
                      row.predicted_q,
                      4
                    )}
                  </div>
                </div>
              {/each}

              {#if !(dashboard.decisions || []).length}
                <div class="tiny muted">
                  Run Portfolio Policy V4 to populate actions.
                </div>
              {/if}
            </div>
          {/if}
        </div>

        <div class="section">
          <div class="section-title">
            Timeline cursor
          </div>

          {#if currentDecision}
            <div
              class="tiny mono"
              style="line-height:1.65"
            >
              <div>
                {currentDecision.date
                  || currentDecision.signal_date
                  || '—'}
              </div>
              <div class="muted">
                {parseTargetAssets(
                  currentDecision
                )}
              </div>

              {#if currentDecision.predicted_q != null}
                <div>
                  predicted Q:
                  <span class="action-q">
                    {fmtNum(
                      currentDecision.predicted_q,
                      6
                    )}
                  </span>
                </div>
              {/if}
            </div>
          {:else}
            <div class="tiny muted">
              No timeline yet.
            </div>
          {/if}
        </div>
      </div>

      <div class="timeline">
        <div class="timeline-head">
          <span>start</span>
          <span>
            {decisions.length
              ? `${timelineIndex + 1}/${decisions.length}`
              : '0/0'}
          </span>
          <span>end</span>
        </div>

        <input
          type="range"
          min="0"
          max={Math.max(
            0,
            decisions.length - 1
          )}
          bind:value={timelineIndex}
          onchange={() => {
            if (
              mode === 'research'
              && currentDecision
            ) {
              loadActionMatrix(
                String(
                  currentDecision.date
                  || currentDecision.signal_date
                  || ''
                ).slice(0, 10)
              );
            }
          }}
        />
      </div>
    </aside>
  </main>

  <section class="bottom-dock">
    <div class="dock-tabs">
      <button
        class:active={dockTab === 'runs'}
        class="dock-tab"
        onclick={() =>
          (dockTab = 'runs')}
      >
        Runs
      </button>

      <button
        class:active={dockTab === 'logs'}
        class="dock-tab"
        onclick={() =>
          (dockTab = 'logs')}
      >
        Logs
      </button>

      <button
        class:active={dockTab === 'actions'}
        class="dock-tab"
        onclick={() =>
          (dockTab = 'actions')}
      >
        Actions
      </button>

      <button
        class:active={dockTab === 'equity'}
        class="dock-tab"
        onclick={() =>
          (dockTab = 'equity')}
      >
        Equity
      </button>

      <button
        class:active={dockTab === 'reports'}
        class="dock-tab"
        onclick={() =>
          (dockTab = 'reports')}
      >
        Artifacts
      </button>

      <div class="dock-spacer"></div>
      <div class="tiny muted">
        Trading terminal × backtest report × experiment tracker
      </div>
    </div>

    <div class="dock-body">
      {#if dockTab === 'runs'}
        <table class="table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Experiment</th>
              <th>Status</th>
              <th>Created</th>
              <th>PID</th>
              <th>Config</th>
            </tr>
          </thead>
          <tbody>
            {#each runs as run}
              <tr
                onclick={() => {
                  selectedRunId =
                    run.id;
                  logLines = [];
                  logCursor = 0;
                }}
                style="cursor:pointer"
              >
                <td class="mono">
                  {run.name}
                </td>
                <td>
                  {run.experiment_id}
                </td>
                <td>
                  <span
                    class={`status ${run.status}`}
                  >
                    {run.status}
                  </span>
                </td>
                <td>
                  {String(
                    run.created_at
                    || ''
                  )
                    .replace(
                      'T',
                      ' '
                    )
                    .slice(0, 19)}
                </td>
                <td>
                  {run.pid || '—'}
                </td>
                <td class="mono">
                  {JSON.stringify(
                    run.config
                  )}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else if dockTab === 'logs'}
        {#if selectedRunId}
          <pre class="log-view">{logLines.join('\n') || 'Waiting for process output…'}</pre>
        {:else}
          <div class="empty">
            Select a run to stream its log.
          </div>
        {/if}
      {:else if dockTab === 'actions'}
        {#if mode === 'research'}
          {#if actionMatrix.rows?.length}
            <table class="table">
              <thead>
                <tr>
                  <th>State</th>
                  <th>Target portfolio</th>
                  <th>Model</th>
                  <th>Pred Q</th>
                  <th>Oracle Q</th>
                  <th>Oracle rank</th>
                  <th>Regret</th>
                  <th>Future return</th>
                  <th>Future MDD</th>
                  <th>Turnover</th>
                </tr>
              </thead>
              <tbody>
                {#each actionMatrix.rows as row}
                  <tr>
                    <td>
                      {row.state_date}
                      {#if row.is_oracle_action}
                        <span
                          class="status completed"
                          style="margin-left:6px"
                        >
                          oracle
                        </span>
                      {/if}
                    </td>
                    <td class="mono">
                      {actionTargetLabel(row)}
                    </td>
                    <td>{row.model}</td>
                    <td class="action-q">
                      {fmtNum(
                        row.predicted_q,
                        6
                      )}
                    </td>
                    <td>
                      {fmtNum(
                        row.oracle_q,
                        6
                      )}
                    </td>
                    <td>
                      #{row.oracle_rank}
                    </td>
                    <td>
                      {fmtNum(
                        row.regret,
                        6
                      )}
                    </td>
                    <td>
                      {fmtPct(
                        row.oracle_terminal_return
                      )}
                    </td>
                    <td>
                      {fmtPct(
                        row.oracle_max_drawdown
                      )}
                    </td>
                    <td>
                      {fmtMoney(
                        row.oracle_turnover
                      )}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <div class="empty">
              Run Portfolio Oracle V4 and Portfolio Student V4 to populate the full action-value matrix.
            </div>
          {/if}
        {:else}
          <table class="table">
            <thead>
              <tr>
                <th>Signal</th>
                <th>Executed</th>
                <th>Your action</th>
                <th>Fees</th>
              </tr>
            </thead>
            <tbody>
              {#each decisions as row}
                <tr>
                  <td>{row.signal_date}</td>
                  <td>{row.execution_date}</td>
                  <td>
                    {parseTargetAssets(
                      row
                    )}
                  </td>
                  <td>
                    {fmtMoney(
                      row.fees
                    )}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      {:else if dockTab === 'equity'}
        <table class="table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Equity</th>
              <th>Cash</th>
              <th>Exposure</th>
              <th>Holdings</th>
            </tr>
          </thead>
          <tbody>
            {#each (mode === 'game' && game ? game.equity_curve || [] : dashboard.equity || []) as row}
              <tr>
                <td>{row.date}</td>
                <td>
                  {fmtMoney(
                    row.equity
                  )}
                </td>
                <td>
                  {fmtMoney(
                    row.cash
                  )}
                </td>
                <td>
                  {fmtPct(
                    row.exposure
                  )}
                </td>
                <td>
                  {row.holdings ?? '—'}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else if dockTab === 'reports'}
        {#if artifactView}
          <div
            class="panel-header"
            style="position:sticky;top:0;z-index:3"
          >
            <button
              class="btn"
              onclick={() =>
                (artifactView = null)}
            >
              ← Back
            </button>
            <div class="panel-title">
              {artifactView.path}
            </div>
            <div class="panel-subtitle">
              {artifactView.kind}
            </div>
          </div>

          {#if artifactView.kind === 'table'}
            <table class="table">
              <thead>
                <tr>
                  {#each artifactView.columns || [] as column}
                    <th>{column}</th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each artifactView.rows || [] as row}
                  <tr>
                    {#each artifactView.columns || [] as column}
                      <td class="mono">
                        {row[column] == null
                          ? '—'
                          : typeof row[column] === 'object'
                            ? JSON.stringify(row[column])
                            : String(row[column])}
                      </td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <pre class="log-view">{artifactView.kind === 'json'
              ? JSON.stringify(artifactView.data, null, 2)
              : artifactView.data || ''}</pre>
          {/if}
        {:else if artifactLoading}
          <div class="empty">
            Loading artifact…
          </div>
        {:else}
          <div class="report-grid">
            {#each reports as report}
              <button
                class="report-card"
                style="text-align:left;color:inherit;cursor:pointer"
                onclick={() =>
                  loadArtifact(
                    report.path
                  )}
              >
                <div class="section-title">
                  {report.kind}
                </div>
                <div class="report-path mono">
                  {report.path}
                </div>
                <div
                  class="tiny muted"
                  style="margin-top:7px"
                >
                  {Math.round(
                    report.size / 1024
                  )} KB
                </div>
              </button>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  </section>
</div>

{#if mode === 'game' && game && holdingModalSymbol && modalPosition}
  <div
    class="trade-modal-backdrop"
    role="presentation"
    onclick={closeHoldingModal}
  >
    <div
      class:profit={modalPosition.unrealized_pnl > 0}
      class:loss={modalPosition.unrealized_pnl < 0}
      class="trade-modal"
      role="dialog"
      aria-modal="true"
      aria-label={`Manage ${modalPosition.symbol} position`}
      onclick={(event) =>
        event.stopPropagation()}
    >
      <div class="trade-modal-head">
        <div>
          <div class="trade-modal-kicker">
            Position manager
          </div>
          <div class="trade-modal-title">
            {modalPosition.symbol}
          </div>
        </div>

        <button
          class="modal-close"
          aria-label="Close"
          onclick={closeHoldingModal}
        >
          ×
        </button>
      </div>

      <div class="modal-pnl-strip">
        <div>
          <span>Unrealized P&L</span>
          <strong>
            {modalPosition.unrealized_pnl >= 0
              ? '+'
              : ''}{fmtMoney(
              modalPosition.unrealized_pnl
            )}
          </strong>
        </div>
        <div>
          <span>Return</span>
          <strong>
            {modalPosition.unrealized_return >= 0
              ? '+'
              : ''}{fmtPct(
              modalPosition.unrealized_return
            )}
          </strong>
        </div>
        <div>
          <span>Weight</span>
          <strong>
            {fmtPct(
              modalPosition.weight
            )}
          </strong>
        </div>
      </div>

      <div class="modal-position-grid">
        <div>
          <span>Quantity</span>
          <strong>
            {modalPosition.quantity}
            shares
          </strong>
        </div>
        <div>
          <span>Average cost</span>
          <strong>
            {fmtMoney(
              modalPosition.average_cost
            )}
          </strong>
        </div>
        <div>
          <span>Current price</span>
          <strong>
            {fmtMoney(
              modalPosition.price
            )}
          </strong>
        </div>
        <div>
          <span>Market value</span>
          <strong>
            {fmtMoney(
              modalPosition.value
            )}
          </strong>
        </div>
        <div>
          <span>Cash available</span>
          <strong>
            {fmtMoney(
              game.cash
            )}
          </strong>
        </div>
        <div>
          <span>Approx max add</span>
          <strong>
            {modalPosition.estimated_max_add_shares}
            shares
          </strong>
        </div>
      </div>

      <div class="modal-divider"></div>

      <div class="modal-order-head">
        <div>
          <div class="section-title">
            Resize position
          </div>
          <div class="tiny muted">
            Orders execute at the next market open.
          </div>
        </div>

        <select
          class="select modal-sizing-select"
          bind:value={orderSizingMode}
        >
          <option value="shares">
            Shares
          </option>
          <option value="rupees">
            ₹ amount
          </option>
          <option value="equity_fraction">
            % of equity
          </option>
          <option value="target_weight">
            Target weight
          </option>
        </select>
      </div>

      {#if orderSizingMode === 'shares'}
        <div class="modal-size-row">
          <button
            class="size-chip"
            onclick={() =>
              (orderQuantity = 1)}
          >
            +1
          </button>
          <button
            class="size-chip"
            onclick={() =>
              (orderQuantity = 5)}
          >
            +5
          </button>
          <button
            class="size-chip"
            onclick={() =>
              (orderQuantity = 10)}
          >
            +10
          </button>
          <button
            class="size-chip"
            onclick={() =>
              (orderQuantity =
                modalPosition.quantity)}
          >
            current qty
          </button>
        </div>

        <div class="control">
          <label>Shares</label>
          <input
            class="input"
            type="number"
            min="1"
            step="1"
            bind:value={orderQuantity}
          />
        </div>
      {:else if orderSizingMode === 'rupees'}
        <div class="control">
          <label>Rupee notional</label>
          <input
            class="input"
            type="number"
            min="1"
            step="1000"
            bind:value={orderAmount}
          />
        </div>
      {:else if orderSizingMode === 'equity_fraction'}
        <div class="control">
          <label>Portfolio equity (%)</label>
          <input
            class="input"
            type="number"
            min="0.01"
            max="100"
            step="1"
            bind:value={orderFractionPct}
          />
        </div>
      {:else}
        <div class="control">
          <label>
            Target weight (%)
          </label>
          <input
            class="input"
            type="number"
            min="0"
            max="100"
            step="1"
            bind:value={orderTargetWeightPct}
          />
        </div>
      {/if}

      <div class="modal-actions">
        {#if orderSizingMode === 'target_weight'}
          <button
            class="btn btn-primary"
            onclick={modalSetTarget}
          >
            Set target weight
          </button>
        {:else}
          <button
            class="btn btn-good"
            onclick={modalAdd}
          >
            Add / average
          </button>
          <button
            class="btn btn-danger"
            onclick={modalTrim}
          >
            Trim / sell
          </button>
        {/if}

        <button
          class="btn btn-danger modal-exit"
          onclick={modalExitPosition}
        >
          Exit full position
        </button>
      </div>
    </div>
  </div>
{/if}

{#if toast}
  <div
    class:error={toast.type === 'error'}
    class="toast"
  >
    {toast.message}
  </div>
{/if}
