<!-- Adapted from freddyaboulton/gradio-range-slider 0.0.8 for Gradio 6. -->
<script lang="ts">
import { Gradio } from "@gradio/utils";
import { Block, BlockTitle } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import type { LoadingStatus } from "@gradio/statustracker";
import { tick } from "svelte";

type RangeProps = { value: [number, number]; minimum: number; maximum: number; step: number; info?: string };
type RangeEvents = { change: undefined; input: undefined; release: number[]; clear_status: LoadingStatus };
let props = $props();
const gradio = new Gradio<RangeEvents, RangeProps>(props);
const initial_value: [number, number] = [...gradio.props.value];
let old_value = $state(JSON.stringify(gradio.props.value));
let disabled = $derived(!gradio.shared.interactive);
let selected_min = $derived(gradio.props.value[0]);
let selected_max = $derived(gradio.props.value[1]);
let rangeLine = $derived(`
    left: ${((selected_min - gradio.props.minimum) / (gradio.props.maximum - gradio.props.minimum)) * 100}%;
    width: ${((selected_max - selected_min) / (gradio.props.maximum - gradio.props.minimum)) * 100}%;
  `);

$effect(() => {
  const current = JSON.stringify(gradio.props.value);
  if (current !== old_value) {
    old_value = current;
    tick().then(() => gradio.dispatch("change"));
  }
});

async function handle_input(index: number, event: Event): Promise<void> {
  const input = event.currentTarget as HTMLInputElement;
  const number = input.valueAsNumber;
  if (!Number.isFinite(number)) {
    input.value = String(gradio.props.value[index]);
    return;
  }
  const value = Math.max(gradio.props.minimum, Math.min(number, gradio.props.maximum));
  gradio.props.value = index === 0 ? [value, Math.max(value, selected_max)] : [Math.min(selected_min, value), value];
  await tick();
  gradio.dispatch("input");
}

async function handle_release(): Promise<void> {
  await tick();
  gradio.dispatch("release", gradio.props.value);
}

async function reset_value(): Promise<void> {
  gradio.props.value = [...initial_value];
  await tick();
  gradio.dispatch("input");
  gradio.dispatch("release", gradio.props.value);
}
</script>

<Block
  visible={gradio.shared.visible}
  elem_id={gradio.shared.elem_id}
  elem_classes={gradio.shared.elem_classes}
  container={gradio.shared.container}
  scale={gradio.shared.scale}
  min_width={gradio.shared.min_width}
>
  <StatusTracker
    autoscroll={gradio.shared.autoscroll}
    i18n={gradio.i18n}
    {...gradio.shared.loading_status}
    on_clear_status={() => gradio.dispatch("clear_status", gradio.shared.loading_status)}
  />

  <div class="wrap">
    <div class="head">
      <BlockTitle show_label={gradio.shared.show_label} info={gradio.props.info}
        >{gradio.shared.label || "Range"}</BlockTitle
      >
      <div class="tab-like-container">
        <input
          aria-label={`min input for ${gradio.shared.label || "Range"}`}
          data-testid="min-input"
          type="number"
          step={gradio.props.step}
          value={selected_min}
          oninput={(event) => handle_input(0, event)}
          min={gradio.props.minimum}
          max={gradio.props.maximum}
          {disabled}
          onchange={handle_release}
          onblur={handle_release}
        >
        <input
          aria-label={`max input for ${gradio.shared.label || "Range"}`}
          data-testid="max-input"
          type="number"
          step={gradio.props.step}
          value={selected_max}
          oninput={(event) => handle_input(1, event)}
          min={gradio.props.minimum}
          max={gradio.props.maximum}
          {disabled}
          onchange={handle_release}
          onblur={handle_release}
        >
        <button class="reset-button" onclick={reset_value} {disabled} aria-label="Reset to default value">↺</button>
      </div>
    </div>
  </div>
  <div class="slider_input_container">
    <span class="min_value">{gradio.props.minimum}</span>
    <div class="range-slider">
      <div class="range-bg"></div>
      <div class="range-line" style={rangeLine} class:disabled={disabled}></div>
      <input
        type="range"
        aria-label={`minimum range for ${gradio.shared.label || "Range"}`}
        {disabled}
        min={gradio.props.minimum}
        max={gradio.props.maximum}
        step={gradio.props.step}
        value={selected_min}
        oninput={(event) => handle_input(0, event)}
        onpointerup={handle_release}
        onkeyup={handle_release}
      >
      <input
        type="range"
        aria-label={`maximum range for ${gradio.shared.label || "Range"}`}
        {disabled}
        min={gradio.props.minimum}
        max={gradio.props.maximum}
        step={gradio.props.step}
        value={selected_max}
        oninput={(event) => handle_input(1, event)}
        onpointerup={handle_release}
        onkeyup={handle_release}
      >
    </div>
    <span class="max_value">{gradio.props.maximum}</span>
  </div>
</Block>

<style>
.wrap {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.head {
  margin-bottom: var(--size-2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
}

.tab-like-container {
  display: flex;
  align-items: stretch;
  border: 1px solid var(--input-border-color);
  border-radius: var(--radius-sm);
  overflow: hidden;
  height: var(--size-6);
}

.slider_input_container {
  display: flex;
  align-items: center;
  gap: var(--size-2);
}

@media (max-width: 420px) {
  .head .tab-like-container {
    margin-bottom: var(--size-4);
  }
}

input[type="number"] {
  border: none;
  border-radius: 0;
  padding: var(--size-1) var(--size-2);
  height: 100%;
  min-width: var(--size-14);
  font-size: var(--text-sm);
  color: var(--body-text-color);
  background: var(--input-background-fill);
}

input[type="number"]:focus {
  box-shadow: none;
  border-width: 2px;
}

.range-slider {
  position: relative;
  width: 100%;
  height: var(--size-4);
}

.range-slider input[type="range"] {
  position: absolute;
  left: 0;
  width: 100%;
  appearance: none;
  outline: none;
  background: transparent;
  pointer-events: none;
  top: 50%;
  transform: translateY(-50%);
  margin: 0;
}
.range-slider input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  height: var(--size-4);
  width: var(--size-4);
  background-color: white;
  border-radius: 50%;
  margin-top: 0;
  box-shadow:
    0 0 0 1px rgba(247, 246, 246, 0.739),
    1px 1px 4px rgba(0, 0, 0, 0.1);
  pointer-events: auto;
  cursor: pointer;
}
/* Uncomment to make slider look the same on firefox */
.range-slider input[type="range"]::-moz-range-thumb {
  height: var(--size-4);
  width: var(--size-4);
  background-color: white;
  border-radius: 50%;
  border: none;
  box-shadow:
    0 0 0 1px rgba(247, 246, 246, 0.739),
    1px 1px 4px rgba(0, 0, 0, 0.1);
  pointer-events: auto;
  cursor: pointer;
}

.range-slider input[type="range"]::-moz-range-track {
  height: var(--size-2);
  background: transparent;
}

.min_value,
.max_value {
  font-size: var(--text-sm);
  color: var(--body-text-color-subdued);
}

.range-line {
  position: absolute;
  left: 0;
  background: var(--slider-color);
  pointer-events: none;
  height: var(--size-2);
  border-radius: var(--radius-xl);
}

.range-bg {
  position: absolute;
  left: 0;
  width: 100%;
  height: var(--size-2);
  z-index: 0;
  background: var(--neutral-200);
  pointer-events: none;
  border-radius: var(--radius-xl);
}

.range-line,
.range-bg {
  top: 50%;
  transform: translateY(-50%);
}

.disabled {
  background-color: var(--body-text-color-subdued);
}

input[type="range"][disabled]::-webkit-slider-thumb {
  cursor: not-allowed;
  background-color: var(--body-text-color-subdued);
}

input[type="range"][disabled]::-moz-range-thumb {
  background-color: var(--body-text-color-subdued);
}

input[type="number"][disabled] {
  cursor: not-allowed;
}

.reset-button {
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  border-left: 1px solid var(--input-border-color);
  cursor: pointer;
  font-size: var(--text-sm);
  color: var(--body-text-color);
  padding: 0 var(--size-2);
  min-width: var(--size-6);
  transition: background-color 0.15s ease-in-out;
}

.reset-button:hover:not(:disabled) {
  background-color: var(--background-fill-secondary);
}

.reset-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
