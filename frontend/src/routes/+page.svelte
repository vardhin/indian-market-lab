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
  let reports = [];
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
  let gameCapital = 50000;
  let gameMaxHoldings = 5;
  let gameSearch = '';
  let gameSelected = 'RELIANCE';
  let switchFrom = '';

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

  async function refreshDashboard() {
    try {
      dashboard = await api.dashboard(
        year
      );
      reports = await api.reports();
      if (dashboard.decisions?.length) {
        timelineIndex =
          dashboard.decisions.length - 1;
      }
    } catch (error) {
      console.error(error);
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
        initial_capital:
          Number(gameCapital),
        max_holdings:
          Number(gameMaxHoldings)
      });

      mode = 'game';
      year = game.year;
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
    if (!game?.id) return;

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
    } catch (error) {
      notify(error.message, 'error');
    }
  }

  function switchAction() {
    if (!switchFrom || !gameSelected) {
      return;
    }

    act(
      'SWITCH',
      {
        from_symbol: switchFrom,
        to_symbol: gameSelected,
        weight:
          1
          / Math.max(
            1,
            gameMaxHoldings
          )
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
              <div class="section-title">
                Market at {game.date}
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
                {#each gameMarket.slice(0, 18) as item}
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
              </div>

              <div class="game-actions">
                <button
                  class="btn btn-good"
                  onclick={() =>
                    act(
                      'BUY',
                      {
                        symbol:
                          gameSelected,
                        weight:
                          1
                          / Math.max(
                            1,
                            gameMaxHoldings
                          )
                      }
                    )}
                >
                  Buy
                </button>

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
                      'SELL',
                      {
                        symbol:
                          gameSelected
                      }
                    )}
                >
                  Sell
                </button>

                <button
                  class="btn btn-danger"
                  onclick={() =>
                    act(
                      'LIQUIDATE'
                    )}
                >
                  Liquidate
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
              {#each game.holdings as holding}
                <div class="holding-row">
                  <span>
                    <strong>
                      {holding.symbol}
                    </strong>
                    · {holding.quantity}
                    @ {holding.price?.toFixed?.(2)}
                  </span>
                  <span>
                    {fmtMoney(
                      holding.value
                    )}
                  </span>
                </div>
              {/each}
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
        <table class="table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Action / target</th>
              <th>Predicted Q</th>
              <th>Fees</th>
            </tr>
          </thead>
          <tbody>
            {#each decisions as row}
              <tr>
                <td>
                  {row.date
                    || row.signal_date}
                </td>
                <td>
                  {parseTargetAssets(
                    row
                  )}
                </td>
                <td>
                  {fmtNum(
                    row.predicted_q,
                    6
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
        <div class="report-grid">
          {#each reports as report}
            <div class="report-card">
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
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </section>
</div>

{#if toast}
  <div
    class:error={toast.type === 'error'}
    class="toast"
  >
    {toast.message}
  </div>
{/if}
