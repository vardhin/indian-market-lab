<script>
  import { onMount } from 'svelte';
  import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';

  let {
    candles = [],
    chartType = 'candles',
    upColor = '#35d07f',
    downColor = '#ff5d6c',
    gridColor = '#1c2330',
    background = '#0b0f15',
    showVolume = true,
    showGrid = true,
    height = 520,
    markers = []
  } = $props();

  let container;
  let chart;
  let priceSeries;
  let volumeSeries;
  let resizeObserver;

  function destroySeries() {
    if (priceSeries && chart) chart.removeSeries(priceSeries);
    if (volumeSeries && chart) chart.removeSeries(volumeSeries);
    priceSeries = null;
    volumeSeries = null;
  }

  function buildSeries() {
    if (!chart) return;
    destroySeries();

    if (chartType === 'line') {
      priceSeries = chart.addLineSeries({
        color: '#8ca8ff',
        lineWidth: 2,
        priceLineVisible: false
      });
      priceSeries.setData(candles.map((d) => ({ time: d.time, value: d.close })));
    } else if (chartType === 'area') {
      priceSeries = chart.addAreaSeries({
        lineColor: '#8ca8ff',
        topColor: 'rgba(102, 132, 255, 0.26)',
        bottomColor: 'rgba(102, 132, 255, 0.015)',
        priceLineVisible: false
      });
      priceSeries.setData(candles.map((d) => ({ time: d.time, value: d.close })));
    } else {
      priceSeries = chart.addCandlestickSeries({
        upColor,
        downColor,
        borderVisible: false,
        wickUpColor: upColor,
        wickDownColor: downColor,
        priceLineVisible: false
      });
      priceSeries.setData(candles.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    }

    if (markers?.length && priceSeries?.setMarkers) {
      priceSeries.setMarkers(markers);
    }

    if (showVolume) {
      volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        color: '#667085'
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 }
      });
      volumeSeries.setData(
        candles
          .filter((d) => d.volume != null)
          .map((d) => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open
              ? 'rgba(53,208,127,.35)'
              : 'rgba(255,93,108,.35)'
          }))
      );
    }

    chart.timeScale().fitContent();
  }

  function applyAppearance() {
    if (!chart) return;
    chart.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: background },
        textColor: '#8f9bb0',
        fontFamily: 'Inter, ui-sans-serif, system-ui'
      },
      grid: {
        vertLines: { color: showGrid ? gridColor : 'transparent' },
        horzLines: { color: showGrid ? gridColor : 'transparent' }
      }
    });
    buildSeries();
  }

  onMount(() => {
    chart = createChart(container, {
      width: container.clientWidth,
      height,
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: background },
        textColor: '#8f9bb0',
        fontFamily: 'Inter, ui-sans-serif, system-ui'
      },
      grid: {
        vertLines: { color: showGrid ? gridColor : 'transparent' },
        horzLines: { color: showGrid ? gridColor : 'transparent' }
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#242b38' },
      timeScale: {
        borderColor: '#242b38',
        timeVisible: true,
        secondsVisible: false
      },
      handleScroll: true,
      handleScale: true
    });

    buildSeries();
    resizeObserver = new ResizeObserver(() => {
      chart?.applyOptions({ width: container.clientWidth });
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver?.disconnect();
      chart?.remove();
    };
  });

  $effect(() => {
    candles;
    chartType;
    upColor;
    downColor;
    showVolume;
    markers;
    if (chart) buildSeries();
  });

  $effect(() => {
    gridColor;
    background;
    showGrid;
    if (chart) applyAppearance();
  });
</script>

<div
  class="market-chart"
  bind:this={container}
  style={`height:${height}px`}
></div>
