package com.phantom.npu;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends Activity {

    private boolean started = false;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private Runnable refresher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btn   = findViewById(R.id.btnToggle);
        TextView log = findViewById(R.id.tvLog);
        TextView st  = findViewById(R.id.tvStatus);

        btn.setOnClickListener(v -> {
            if (!started) {
                startForegroundService(new Intent(this, DaemonService.class));
                btn.setText("Stop Daemon");
                started = true;
            } else {
                Intent stop = new Intent(this, DaemonService.class);
                stop.setAction("STOP");
                startService(stop);
                btn.setText("Start Daemon");
                started = false;
            }
        });

        refresher = () -> {
            st.setText(DaemonService.lastStatus);
            log.setText(DaemonService.lastLog);
            handler.postDelayed(refresher, 500);
        };
        handler.post(refresher);
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacks(refresher);
        super.onDestroy();
    }
}
