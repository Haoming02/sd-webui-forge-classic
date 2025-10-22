(ns hello
  (:require ["vscode" :as vscode]))

(defn greet []
  (vscode/window.showInformationMessage "Hello from custom Joyride script!"))