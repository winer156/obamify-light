use eframe::wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use web_sys::DedicatedWorkerGlobalScope;
use web_sys::js_sys;

#[derive(Serialize, Deserialize)]
pub enum WorkerReq {
    Process {
        source: crate::app::preset::UnprocessedPreset,
        settings: super::GenerationSettings,
    },
}

use crate::app::calculate::ProgressMsg;
use crate::app::calculate::process;

// thread_local! {
//     static CANCELLED: Rc<Cell<bool>> = Rc::new(Cell::new(false));
// }

#[wasm_bindgen]
pub fn worker_entry() {
    let global: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let global_for_handler = global.clone();

    let handler = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        // Deserialize the incoming request
        let req: WorkerReq = match serde_wasm_bindgen::from_value(e.data()) {
            Ok(v) => v,
            Err(err) => {
                let _ = global_for_handler.post_message(
                    &serde_wasm_bindgen::to_value(&ProgressMsg::Error(format!("bad req: {err}")))
                        .unwrap(),
                );
                return;
            }
        };

        match req {
            WorkerReq::Process { source, settings } => {
                // Run job; if you need to keep the UI responsive in the worker,
                // wrap in an async task and yield occasionally.
                let global2 = global_for_handler.clone();

                // progress sink -> postMessage
                let mut sink = |msg: ProgressMsg| {
                    let _ = global2.post_message(&serde_wasm_bindgen::to_value(&msg).unwrap());
                };

                // If you need to yield, you can insert tiny awaits between steps.
                // Here we just call the portable sync fn:
                if let Err(e) = process(source, settings, &mut sink) {
                    sink(ProgressMsg::Error(e.to_string()));
                }
            }
        }
    }) as Box<dyn FnMut(_)>);

    global.set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();
}
