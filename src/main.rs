#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 1024.0])
            .with_min_inner_size([400.0, 400.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon128.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "obamify",
        native_options,
        Box::new(|cc| Ok(Box::new(obamify::ObamifyApp::new(cc)))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn start_app() {
    use eframe::wasm_bindgen::JsCast as _;
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    //web_sys::console::log_1(&"Starting obamify...".into());

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Warn).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(obamify::ObamifyApp::new(cc)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text
                        .set_inner_html(&format!("<p> The app has crashed. Error: {e:#?} </p>"));
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
pub fn main() {
    use wasm_bindgen::JsCast as _;
    console_error_panic_hook::set_once();

    // If we have a Window, we’re on the page → run the app.
    if web_sys::window().is_some() {
        start_app();
        return;
    }

    // Otherwise, if we have a DedicatedWorkerGlobalScope, we’re in a worker → install worker.
    if web_sys::js_sys::global()
        .dyn_ref::<web_sys::DedicatedWorkerGlobalScope>()
        .is_some()
    {
        obamify::worker_entry(); // <- your existing function that sets onmessage, etc.
        return;
    }

    // Fallback: unknown environment
    web_sys::console::warn_1(
        &"Unknown global (not Window / not DedicatedWorkerGlobalScope)".into(),
    );
}
