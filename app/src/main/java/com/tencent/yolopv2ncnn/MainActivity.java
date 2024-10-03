// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.yolopv2ncnn;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.PopupMenu;
import android.widget.SeekBar;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    public static final int REQUEST_CAMERA = 100;

    private SurfaceView cameraView;
    private ImageButton btnMenu;
    private PopupMenu popupMenu;
    private final Yolopv2Ncnn yolopv2ncnn = new Yolopv2Ncnn();
    private SharedPreferences sharedPreferences;

    private float currentZoom = 1.0f;
    private static final float MIN_ZOOM = 1.0f;
    private static final float MAX_ZOOM = 3.0f;

    private ExecutorService executor = Executors.newSingleThreadExecutor();
    private Handler handler = new Handler(Looper.getMainLooper());

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        sharedPreferences = getPreferences(MODE_PRIVATE);

        cameraView = (SurfaceView) findViewById(R.id.view_camera);
        btnMenu = (ImageButton) findViewById(R.id.btn_menu);

        setupMenu();
        setupCameraView();

        loadSettings();

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        initializeModel();
    }

    private void initializeModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                final boolean result = yolopv2ncnn.loadModel(getAssets(), getCoreTypeFromSettings());
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (!result) {
                            showErrorDialog("Failed to load model");
                        }
                    }
                });
            }
        });
    }

    private void setupMenu() {
        popupMenu = new PopupMenu(this, btnMenu);
        popupMenu.getMenuInflater().inflate(R.menu.menu_main, popupMenu.getMenu());

        popupMenu.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                return MainActivity.this.onMenuItemClick(item);
            }
        });

        btnMenu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                popupMenu.show();
            }
        });
    }

    private boolean onMenuItemClick(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_cpu:
                updateCoreType(0);
                return true;
            case R.id.menu_gpu:
                updateCoreType(1);
                return true;
            case R.id.menu_drivable:
                updateDrivableArea(!item.isChecked());
                item.setChecked(!item.isChecked());
                return true;
            case R.id.menu_lane:
                updateLaneDetection(!item.isChecked());
                item.setChecked(!item.isChecked());
                return true;
            case R.id.menu_detection:
                updateObjectDetection(!item.isChecked());
                item.setChecked(!item.isChecked());
                return true;
            case R.id.menu_zoom:
                showZoomDialog();
                return true;
        }
        return false;
    }

    private void updateCoreType(final int coreType) {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                final boolean result = yolopv2ncnn.loadModel(getAssets(), coreType);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (!result) {
                            showErrorDialog("Failed to update core type");
                        }
                    }
                });
            }
        });
        saveSettings("core", coreType);
    }

    private void updateDrivableArea(boolean enable) {
        yolopv2ncnn.enableDrivableArea(enable);
        saveSettings("drivable", enable);
    }

    private void updateLaneDetection(boolean enable) {
        yolopv2ncnn.enableLaneDetection(enable);
        saveSettings("lane", enable);
    }

    private void updateObjectDetection(boolean enable) {
        yolopv2ncnn.enableObjectDetection(enable);
        saveSettings("detection", enable);
    }

    private void setupCameraView() {
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {}

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                yolopv2ncnn.setOutputWindow(holder.getSurface());
            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {}
        });
    }

    private void showZoomDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        LayoutInflater inflater = getLayoutInflater();
        View dialogView = inflater.inflate(R.layout.zoom_dialog, null);
        builder.setView(dialogView);

        final SeekBar seekBar = (SeekBar) dialogView.findViewById(R.id.seekbar_zoom);
        int progress = (int) ((currentZoom - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM) * 200);
        seekBar.setProgress(progress);

        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    currentZoom = MIN_ZOOM + (MAX_ZOOM - MIN_ZOOM) * (progress / 200f);
                    updateZoom();
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
            }
        });

        builder.create().show();
    }

    private void updateZoom() {
        yolopv2ncnn.setZoom(currentZoom);
        saveSettings("zoom", currentZoom);
    }

    private void loadSettings() {
        int core = sharedPreferences.getInt("core", 0);
        updateCoreType(core);

        boolean isDrivable = sharedPreferences.getBoolean("drivable", true);
        updateDrivableArea(isDrivable);
        popupMenu.getMenu().findItem(R.id.menu_drivable).setChecked(isDrivable);

        boolean isLane = sharedPreferences.getBoolean("lane", true);
        updateLaneDetection(isLane);
        popupMenu.getMenu().findItem(R.id.menu_lane).setChecked(isLane);

        boolean isDetection = sharedPreferences.getBoolean("detection", true);
        updateObjectDetection(isDetection);
        popupMenu.getMenu().findItem(R.id.menu_detection).setChecked(isDetection);

        currentZoom = sharedPreferences.getFloat("zoom", 1f);
        updateZoom();
    }

    private int getCoreTypeFromSettings() {
        return sharedPreferences.getInt("core", 0);
    }

    private void saveSettings(String key, int value) {
        sharedPreferences.edit().putInt(key, value).apply();
    }

    private void saveSettings(String key, boolean value) {
        sharedPreferences.edit().putBoolean(key, value).apply();
    }

    private void saveSettings(String key, float value) {
        sharedPreferences.edit().putFloat(key, value).apply();
    }

    private void showErrorDialog(String message) {
        new AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        } else {
            yolopv2ncnn.openCamera();
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        yolopv2ncnn.closeCamera();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
    }
}