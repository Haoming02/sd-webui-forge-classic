import { build } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { readFileSync, readdirSync } from "node:fs";

// Match Gradio 6's custom-component build, including its isolated Svelte runtime.
await build({
  configFile: false,
  plugins: [
    svelte({ compilerOptions: { discloseVersion: false } }),
    {
      name: "gradio-custom-context",
      transform: (code) => code.replace('"_NORMAL_"', '"_CC_"'),
    },
    {
      name: "bundled-licenses",
      generateBundle(options, bundle) {
        const licenses = new Map();
        for (const chunk of Object.values(bundle)) {
          if (chunk.type !== "chunk") continue;
          for (const id of Object.keys(chunk.modules)) {
            if (id.startsWith("\0")) continue;
            const directory = id.replaceAll("\\", "/").match(/^(.*\/node_modules\/(?:@[^/]+\/)?[^/]+)/)?.[1];
            if (!directory || licenses.has(directory)) continue;
            const pkg = JSON.parse(readFileSync(`${directory}/package.json`, "utf8"));
            const filename = readdirSync(directory).find((name) => /^(license|licence|copying)(\..*)?$/i.test(name));
            if (!filename) throw new Error(`Missing license for bundled package ${pkg.name}`);
            licenses.set(
              directory,
              `${pkg.name}@${pkg.version}\n\n${readFileSync(`${directory}/${filename}`, "utf8")}`,
            );
          }
        }
        this.emitFile({
          type: "asset",
          fileName: "THIRD_PARTY_LICENSES.txt",
          source: [...licenses.values()].sort().join("\n\n---\n\n"),
        });
      },
    },
  ],
  build: {
    outDir: "../templates/component",
    emptyOutDir: true,
    lib: {
      entry: ["Index.svelte", "svelte_runtime_entry.js"],
      formats: ["es"],
    },
    rollupOptions: {
      output: {
        entryFileNames: (chunk) => (chunk.name === "svelte_runtime_entry" ? "svelte_runtime_entry.js" : "index.js"),
        assetFileNames: (asset) => (asset.names[0].endsWith(".css") ? "style.css" : asset.names[0]),
      },
    },
  },
});
