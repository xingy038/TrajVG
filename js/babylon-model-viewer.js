class BabylonModelViewer {
    constructor(canvas) {
        this.canvas = canvas;
        this.isLoading = false; // Initialize loading state to prevent concurrent loads

        // Enable mobile-friendly engine options
        const engineOptions = {
            preserveDrawingBuffer: true,
            stencil: true,
            antialias: true,
            alpha: false,
            failIfMajorPerformanceCaveat: false
        };
        this.engine = new BABYLON.Engine(canvas, true, engineOptions);
        this.scene = new BABYLON.Scene(this.engine);
        this.scene.clearColor = new BABYLON.Color4(1.0, 1.0, 1.0, 1.0);

        this.camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 2, Math.PI / 2, 10, BABYLON.Vector3.Zero(), this.scene);

        // Enhanced mobile-friendly camera setup
        this.camera.attachControl(canvas, true);
        this.camera.inputs.attached.pointers.buttons = [0, 1, 2]; // Support all mouse buttons
        this.camera.inputs.attached.pointers.angularSensibilityX = 1000;
        this.camera.inputs.attached.pointers.angularSensibilityY = 1000;
        this.camera.inputs.attached.pointers.panningSensibility = 1000;

        // Enable touch support explicitly for mobile
        if (this.camera.inputs.attached.pointers) {
            this.camera.inputs.attached.pointers.multiTouchPanning = true;
            this.camera.inputs.attached.pointers.multiTouchPanAndZoom = true;
        }
        this.scene.ambientColor = new BABYLON.Color3(1.0, 1.0, 1.0);

        this.light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 0, 1), this.scene);
        this.light.intensity = 4;

        this._PointerDownPos = null;

        this.canvas.addEventListener("pointerdown", (evt) => {
            if (evt.button === 0) {
                this._PointerDownPos = { x: this.scene.pointerX, y: this.scene.pointerY };
            }
        });

        this.canvas.addEventListener("pointerup", (evt) => {
            if (evt.button === 0 && this._PointerDownPos) {
                const dx = this.scene.pointerX - this._PointerDownPos.x;
                const dy = this.scene.pointerY - this._PointerDownPos.y;
                const distanceSquared = dx * dx + dy * dy;

                const clickThreshold = 3 * 3;

                if (distanceSquared < clickThreshold) {
                    const pickResult = this.scene.pick(this.scene.pointerX, this.scene.pointerY);

                    if (pickResult.hit && this.onClickPick) {
                        this.onClickPick(pickResult);
                    }
                }
                this._PointerDownPos = null;
            }
        });

        this.engine.runRenderLoop(() => {
            this.scene.render();
        });

        window.addEventListener("resize", () => {
            this.engine.resize();
        });

        this.canvas.addEventListener("wheel", function (event) {
            event.preventDefault();
        }, { passive: false });

        canvas.addEventListener("contextmenu", (evt) => evt.preventDefault());

        // Add comprehensive mobile touch event handling
        this.setupMobileTouchEvents(canvas);
    }

    setupMobileTouchEvents(canvas) {
        // Prevent default touch behaviors that interfere with 3D interaction
        canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
        }, { passive: false });

        canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
        }, { passive: false });

        canvas.addEventListener("touchend", (e) => {
            e.preventDefault();
        }, { passive: false });

        // Additional mobile-specific setup
        if (typeof window !== 'undefined' && 'ontouchstart' in window) {
            // Mobile device detected - configure for better touch handling
            if (this.camera.inputs.attached.pointers) {
                const pointerInput = this.camera.inputs.attached.pointers;

                // Optimize for touch
                pointerInput.angularSensibilityX = 500;  // Make rotation more responsive on mobile
                pointerInput.angularSensibilityY = 500;
                pointerInput.panningSensibility = 500;   // Make panning more responsive

                // Enable multi-touch gestures
                pointerInput.multiTouchPanning = true;
                pointerInput.multiTouchPanAndZoom = true;
                pointerInput.pinchPrecision = 50;        // Improve pinch-to-zoom sensitivity
                pointerInput.pinchDeltaPercentage = 0.01; // Fine-tune pinch responsiveness
            }
        }
    }

    clearMeshes() {
        this.scene.meshes.forEach(mesh => {
            if (!(mesh instanceof BABYLON.Camera) && !(mesh instanceof BABYLON.Light)) {
                mesh.dispose();
            }
        });
    }

    clearMaterials() {
        this.scene.materials.forEach(mat => mat.dispose());
    }

    loadGLB(fileUrl, onSuccess, onError) {
        // Clear previous meshes and materials
        this.clearMeshes();
        this.clearMaterials();


        // Load the glb file
        BABYLON.SceneLoader.Append(
            "./",
            fileUrl,
            this.scene,
            () => {
                if (onSuccess) onSuccess();
            },
            null, // onProgress callback
            (scene, message, exception) => {
                console.error('BabylonModelViewer: Failed to load GLB:', fileUrl, 'Message:', message, 'Exception:', exception);
                if (onError) onError({ message, exception });
            }
        );
    }
}


window.addEventListener("DOMContentLoaded", () => {
    const modelViewers = document.querySelectorAll('.babylon-model-viewer');

    modelViewers.forEach((canvasElement) => {
        canvasElement.viewer = new BabylonModelViewer(canvasElement);
    });
});
