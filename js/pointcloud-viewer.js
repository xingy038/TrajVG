function computeSceneZGeometricMean(scene) {
    let logZSum = 0;
    let count = 0;

    for (const mesh of scene.meshes) {
        if (!mesh.isVisible || !mesh.getVerticesData || !mesh.getTotalVertices()) continue;
        mesh.computeWorldMatrix(true);
        const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
        if (!positions) continue;

        const matrix = mesh.getWorldMatrix();

        for (let i = 0; i < positions.length; i += 3) {
            const pos = BABYLON.Vector3.TransformCoordinates(
                BABYLON.Vector3.FromArray(positions, i),
                matrix
            );
            if (pos.z < 0) {
                logZSum += Math.log(-pos.z);
                count++;
            }
        }
    }

    if (count === 0) {
        console.warn('computeSceneZGeometricMean: No valid vertices found, using fallback distance');
        return 5.0;
    }

    return Math.exp(logZSum / count);
}

function resetViewer(viewer) {
    const bounds = computeSceneBounds(viewer.scene);
    if (bounds) {
        frameViewerToBounds(viewer, bounds);
    } else {
        const zGeoMean = computeSceneZGeometricMean(viewer.scene);
        const center = new BABYLON.Vector3(0, 0, -zGeoMean);
        viewer.camera.setTarget(center);

        const sceneSize = 3 * zGeoMean;
        viewer.sceneSize = sceneSize;
        viewer.camera.radius = sceneSize / 2;
        viewer.camera.lowerRadiusLimit = 0.3 * sceneSize;
        viewer.camera.upperRadiusLimit = sceneSize * 2;
        viewer.camera.wheelPrecision = 500 / sceneSize;

        viewer.camera.angularSensibilityX = 2000;
        viewer.camera.angularSensibilityY = 2000;
        viewer.camera.panningSensibility = 2000;
        viewer.camera.inertia = 0.9;

        viewer.camera.alpha = 0.5 * Math.PI;
        viewer.camera.beta = 0.4 * Math.PI;
    }
}

function computeSceneBounds(scene) {
    let min = new BABYLON.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY);
    let max = new BABYLON.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY);
    let hasMesh = false;

    scene.meshes.forEach(mesh => {
        if (!mesh.isVisible || !mesh.getBoundingInfo || !mesh.getTotalVertices()) return;
        mesh.computeWorldMatrix(true);
        const box = mesh.getBoundingInfo().boundingBox;
        min = BABYLON.Vector3.Minimize(min, box.minimumWorld);
        max = BABYLON.Vector3.Maximize(max, box.maximumWorld);
        hasMesh = true;
    });

    if (!hasMesh) return null;
    return { min, max };
}

function frameViewerToBounds(viewer, bounds) {
    const center = bounds.min.add(bounds.max).scale(0.5);
    const size = bounds.max.subtract(bounds.min);
    const radius = Math.max(size.x, size.y, size.z) * 0.6;

    viewer.sceneSize = radius * 2;
    viewer.camera.setTarget(center);
    viewer.camera.radius = radius > 0 ? radius : 5;
    viewer.camera.lowerRadiusLimit = viewer.camera.radius * 0.2;
    viewer.camera.upperRadiusLimit = viewer.camera.radius * 5;
    viewer.camera.wheelPrecision = 500 / viewer.camera.radius;

    viewer.camera.angularSensibilityX = 2000;
    viewer.camera.angularSensibilityY = 2000;
    viewer.camera.panningSensibility = 2000;
    viewer.camera.inertia = 0.9;
    viewer.camera.allowUpsideDown = true;
    viewer.camera.lowerBetaLimit = null;
    viewer.camera.upperBetaLimit = null;

    // Slightly above-front view
    viewer.camera.alpha = 0.5 * Math.PI;
    viewer.camera.beta = 0.35 * Math.PI;
}

function initPointCloudViewer() {
    const canvas = document.getElementById('renderCanvas');
    if (!canvas || !canvas.viewer) return;

    const viewer = canvas.viewer;
    const framePrevBtn = document.getElementById('framePrevBtn');
    const frameNextBtn = document.getElementById('frameNextBtn');
    const frameLabel = document.getElementById('frameLabel');

    if (window.location.protocol === 'file:') {
        if (frameLabel) {
            frameLabel.innerHTML =
                '3D viewer needs an HTTP server. Run <code>python -m http.server 8000</code> and open ' +
                '<code>http://localhost:8000</code>.';
        }
        return;
    }

    const glbPath = getActiveThumbnailGlb() || 'assets/pointclouds/skating.glb';
    loadPointCloud(viewer, glbPath, framePrevBtn, frameNextBtn, frameLabel);
    initSceneThumbnails(viewer, framePrevBtn, frameNextBtn, frameLabel);
}

window.addEventListener('DOMContentLoaded', initPointCloudViewer);

function setupFrameControls(viewer, prevBtn, nextBtn, labelEl) {
    const scene = viewer.scene;
    const frameMeshes = scene.meshes
        .filter(mesh => mesh.name && mesh.name.startsWith('frame_'))
        .sort((a, b) => a.name.localeCompare(b.name));

    const cameraSources = getCameraSources(scene);
    const frustums = [];
    const frameCamera = ensureFrameCamera(viewer, scene);
    attachWheelZoom(viewer, frameCamera);
    attachPanControls(viewer, frameCamera);
    configureFreeNavigation(frameCamera);

    let displayMode = 'all'; // current | past | all

    const sceneBounds = computeSceneBounds(scene);
    if (sceneBounds) {
        viewer.sceneSize = sceneBounds.max.subtract(sceneBounds.min).length();
        viewer.sceneCenter = sceneBounds.min.add(sceneBounds.max).scale(0.5);
    }

    if (frameMeshes.length === 0) {
        resetViewer(viewer);
        if (labelEl) labelEl.textContent = 'Frame 0 / 0';
        return;
    }

    let currentIndex = 0;

    const updateVisibility = () => {
        frameMeshes.forEach((mesh, i) => {
            if (displayMode === 'all') {
                mesh.isVisible = true;
            } else if (displayMode === 'past') {
                mesh.isVisible = i <= currentIndex;
            } else {
                mesh.isVisible = i === currentIndex;
            }
        });

    };

    const setFrame = (index) => {
        currentIndex = Math.max(0, Math.min(frameMeshes.length - 1, index));
        updateVisibility();

        applyFrameCamera(frameCamera, cameraSources[currentIndex], viewer, scene);

        if (labelEl) {
            labelEl.textContent = `Frame ${currentIndex + 1} / ${frameMeshes.length}`;
        }
        if (prevBtn) prevBtn.disabled = currentIndex === 0;
        if (nextBtn) nextBtn.disabled = currentIndex === frameMeshes.length - 1;
    };

    if (prevBtn) prevBtn.onclick = () => setFrame(currentIndex - 1);
    if (nextBtn) nextBtn.onclick = () => setFrame(currentIndex + 1);

    const initialIndex = frameMeshes.length <= 2 ? 0 : Math.floor(frameMeshes.length / 2);
    setFrame(initialIndex);
}

function loadPointCloud(viewer, glbPath, prevBtn, nextBtn, labelEl) {
    viewer.isLoading = true;
    viewer.clearMeshes();
    viewer.clearMaterials();

    viewer.loadGLB(glbPath, () => {
        setupFrameControls(viewer, prevBtn, nextBtn, labelEl);
        viewer.isLoading = false;
    }, (error) => {
        console.error('Failed to load GLB:', glbPath, error);
        if (labelEl) labelEl.textContent = 'Failed to load point cloud data.';
        viewer.isLoading = false;
    });
}

function getActiveThumbnailGlb() {
    const active = document.querySelector('.scene-thumbnail-bar .scene-thumbnail.active')
        || document.querySelector('.scene-thumbnail-bar .scene-thumbnail');
    if (!active) return null;
    return active.dataset.glb || active.getAttribute('data-glb');
}

function initSceneThumbnails(viewer, prevBtn, nextBtn, labelEl) {
    const thumbs = Array.from(document.querySelectorAll('.scene-thumbnail-bar .scene-thumbnail'));
    if (thumbs.length === 0) return;

    thumbs.forEach((thumb) => {
        thumb.addEventListener('click', () => {
            const glbPath = thumb.dataset.glb || thumb.getAttribute('data-glb');
            if (!glbPath) return;
            thumbs.forEach(t => t.classList.remove('active'));
            thumb.classList.add('active');
            loadPointCloud(viewer, glbPath, prevBtn, nextBtn, labelEl);
        });
    });
}

function applyFrameCamera(frameCamera, frameCam, viewer, scene) {
    if (frameCam) {
        if (scene.activeCamera && scene.activeCamera !== frameCamera) {
            scene.activeCamera.detachControl();
        }
        scene.activeCamera = frameCamera;
        frameCamera.attachControl(viewer.canvas, true);

        const camPos = frameCam.getPosition();
        const forward = frameCam.getForward();
        const up = frameCam.getUp();
        const zoomOut = Math.max((viewer.sceneSize || 1) * 0.25, 0.5);
        frameCamera.position.copyFrom(camPos.subtract(forward.scale(zoomOut)));
        frameCamera.upVector.copyFrom(up);
        frameCamera.setTarget(camPos.add(forward));
        frameCamera.minZ = 0.001;
        frameCamera.maxZ = 1000000;
        if (frameCam.node && frameCam.node.fov) {
            frameCamera.fov = frameCam.node.fov;
        }
    } else {
        if (scene.activeCamera && scene.activeCamera !== viewer.camera) {
            scene.activeCamera.detachControl();
        }
        scene.activeCamera = viewer.camera;
        viewer.camera.attachControl(viewer.canvas, true);
        resetViewer(viewer);
    }
}

function configureFreeNavigation(camera) {
    if (!camera) return;
    camera.inertia = 0.7;
    if (typeof camera.angularSensibility === 'number') {
        camera.angularSensibility = 600;
    }
    if (camera.inputs && camera.inputs.attached && camera.inputs.attached.pointers) {
        const pointers = camera.inputs.attached.pointers;
        if (typeof pointers.angularSensibilityX === 'number') pointers.angularSensibilityX = 600;
        if (typeof pointers.angularSensibilityY === 'number') pointers.angularSensibilityY = 600;
    }
}

function attachPanControls(viewer, frameCamera) {
    if (!viewer || !viewer.canvas || viewer._framePanHandlers) return;

    let panning = false;
    let lastX = 0;
    let lastY = 0;
    let controlsDetached = false;

    const detachControls = () => {
        if (controlsDetached) return;
        try { frameCamera.detachControl(); } catch (e) {}
        controlsDetached = true;
    };

    const restoreControls = () => {
        if (!controlsDetached) return;
        try { frameCamera.attachControl(viewer.canvas, true); } catch (e) {}
        controlsDetached = false;
    };

    const onPointerDown = (evt) => {
        const isShiftLeft = evt.shiftKey && evt.button === 0;
        const isMiddleRight = evt.button === 1 || evt.button === 2;
        if (!isShiftLeft && !isMiddleRight) return;
        if (viewer.scene && viewer.scene.activeCamera !== frameCamera) return;
        detachControls();
        panning = true;
        lastX = evt.clientX;
        lastY = evt.clientY;
        evt.preventDefault();
        evt.stopImmediatePropagation();
    };

    const onPointerMove = (evt) => {
        if (!panning) return;
        if (viewer.scene && viewer.scene.activeCamera !== frameCamera) return;
        const dx = evt.clientX - lastX;
        const dy = evt.clientY - lastY;
        lastX = evt.clientX;
        lastY = evt.clientY;

        const scale = (viewer.sceneSize || 1) * 0.0005;
        const right = frameCamera.getDirection(BABYLON.Axis.X);
        const up = frameCamera.getDirection(BABYLON.Axis.Y);
        frameCamera.position.addInPlace(right.scale(-dx * scale));
        frameCamera.position.addInPlace(up.scale(dy * scale));
        evt.preventDefault();
        evt.stopImmediatePropagation();
    };

    const onPointerUp = () => {
        panning = false;
        restoreControls();
    };

    viewer._framePanHandlers = { onPointerDown, onPointerMove, onPointerUp };
    viewer.canvas.addEventListener('pointerdown', onPointerDown);
    viewer.canvas.addEventListener('pointermove', onPointerMove);
    viewer.canvas.addEventListener('pointerup', onPointerUp);
    viewer.canvas.addEventListener('pointerleave', onPointerUp);
}

function getCameraSources(scene) {
    const cameras = scene.cameras
        .filter(cam => cam.name && cam.name.startsWith('camera_'))
        .sort((a, b) => a.name.localeCompare(b.name));

    if (cameras.length > 0) {
        return cameras.map(cam => ({
            name: cam.name,
            node: cam,
            getPosition: () => cam.getAbsolutePosition().clone(),
            getForward: () => cam.getForwardRay(1).direction.clone(),
            getUp: () => {
                const up = new BABYLON.Vector3(0, 1, 0);
                BABYLON.Vector3.TransformNormalToRef(up, cam.getWorldMatrix(), up);
                up.normalize();
                return up;
            },
            getWorldMatrix: () => cam.getWorldMatrix(),
        }));
    }

    const nodes = scene.transformNodes
        .filter(node => node.name && node.name.startsWith('camera_'))
        .sort((a, b) => a.name.localeCompare(b.name));

    if (nodes.length === 0) {
        console.warn('No camera nodes found in scene.');
    }

    return nodes.map(node => ({
        name: node.name,
        node,
        getPosition: () => node.getAbsolutePosition().clone(),
        getForward: () => {
            const forward = new BABYLON.Vector3(0, 0, -1);
            const dir = BABYLON.Vector3.TransformNormal(forward, node.getWorldMatrix());
            dir.normalize();
            return dir;
        },
        getUp: () => {
            const up = new BABYLON.Vector3(0, 1, 0);
            const dir = BABYLON.Vector3.TransformNormal(up, node.getWorldMatrix());
            dir.normalize();
            return dir;
        },
        getWorldMatrix: () => node.getWorldMatrix(),
    }));
}

function ensureFrameCamera(viewer, scene) {
    if (viewer.frameCamera) return viewer.frameCamera;
    const cam = new BABYLON.UniversalCamera('frameCamera', new BABYLON.Vector3(0, 0, 0), scene);
    cam.minZ = 0.001;
    cam.maxZ = 1000000;
    viewer.frameCamera = cam;
    return cam;
}

function attachWheelZoom(viewer, frameCamera) {
    if (!viewer || !viewer.canvas || viewer._frameWheelHandler) return;

    const handler = (evt) => {
        const scene = viewer.scene;
        if (!scene || scene.activeCamera !== frameCamera) return;

        const delta = Number.isFinite(evt.deltaY) ? evt.deltaY : 0;
        if (delta === 0) return;

        const scale = typeof viewer.sceneSize === 'number' && viewer.sceneSize > 0 ? viewer.sceneSize : 1.0;
        const step = -delta * 0.0001 * scale;
        const forward = frameCamera.getForwardRay(1).direction;
        frameCamera.position.addInPlace(forward.scale(step));

        evt.preventDefault();
    };

    viewer._frameWheelHandler = handler;
    viewer.canvas.addEventListener('wheel', handler, { passive: false });
}

function createCameraFrustums(scene, cameraSources, frameMeshes) {
    const frustums = [];
    const engine = scene.getEngine();
    const aspect = engine ? engine.getAspectRatio(scene.activeCamera || scene.cameras[0] || {}) : 1.6;
    let size = 0.15;
    const bounds = frameMeshes && frameMeshes.length > 0 ? computeSceneBounds(scene) : null;
    if (bounds) {
        const diag = bounds.max.subtract(bounds.min).length();
        size = Math.max(diag * 0.02, 0.01);
    }
    cameraSources.forEach((camSource) => {
        const cam = camSource.node;
        const fov = cam && cam.fov ? cam.fov : (60 * Math.PI / 180);
        const halfHeight = Math.tan(fov / 2) * size;
        const halfWidth = halfHeight * aspect;
        const z = -size;

        const corners = [
            new BABYLON.Vector3(-halfWidth, -halfHeight, z),
            new BABYLON.Vector3(halfWidth, -halfHeight, z),
            new BABYLON.Vector3(halfWidth, halfHeight, z),
            new BABYLON.Vector3(-halfWidth, halfHeight, z),
        ];

        const lines = [
            [BABYLON.Vector3.Zero(), corners[0]],
            [BABYLON.Vector3.Zero(), corners[1]],
            [BABYLON.Vector3.Zero(), corners[2]],
            [BABYLON.Vector3.Zero(), corners[3]],
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
        ];

        if (cam && cam.computeWorldMatrix) {
            cam.computeWorldMatrix(true);
        }
        const wm = camSource.getWorldMatrix();
        const scale = new BABYLON.Vector3();
        const rotation = new BABYLON.Quaternion();
        const translation = new BABYLON.Vector3();
        wm.decompose(scale, rotation, translation);

        const frustum = BABYLON.MeshBuilder.CreateLineSystem(`frustum_${camSource.name}`, { lines }, scene);
        frustum.color = new BABYLON.Color3(0.1, 0.6, 0.2);
        frustum.isPickable = false;
        frustum.position = translation;
        frustum.rotationQuaternion = rotation;
        frustum.scaling = scale;
        frustum.isVisible = false;

        frustums.push(frustum);
    });

    return frustums;
}
