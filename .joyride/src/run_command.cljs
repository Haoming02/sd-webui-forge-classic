(ns run_command
  (:require ["vscode" :as vscode]))

(defn execute-command [cmd & args]
  (apply vscode/commands.executeCommand cmd args))

(defn open-file [path]
  (let [uri (vscode/Uri.file path)]
    (vscode/window.showTextDocument uri)))