package com.phantom.npu;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.LiteRtEnvironment;
import com.google.ai.edge.litert.LiteRtOptions;
import com.google.ai.edge.litert.Accelerator;

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.*;

public class DaemonService extends Service {

    private static final String TAG = "PhantomNPU";
    private static final int PORT = 50051;
    private static final int CHANNEL_ID_INT = 1;
    private static final String CHANNEL_ID = "phantom_npu";

    // Protocol
    static final int MSG_PING   = 0x01;
    static final int MSG_LOAD   = 0x03;
    static final int MSG_INFER  = 0x04;
    static final int MSG_UNLOAD = 0x05;
    static final int MSG_BENCH  = 0x06;
    static final int MSG_ERROR  = 0xFF;
    static final int MSG_PONG   = 0x81;
    static final int MSG_ACK    = 0x82;
    static final int MSG_RESULT = 0x83;

    private volatile boolean running = false;
    private ServerSocket serverSocket;
    private final ExecutorService pool = Executors.newCachedThreadPool();
    private final Map<Integer, SessionState> sessions = new ConcurrentHashMap<>();
    private int nextSessionId = 1;

    public static volatile String lastStatus = "Stopped";
    public static volatile String lastLog = "";

    // ── session ──────────────────────────────────────────────
    static class SessionState {
        int id;
        CompiledModel model;
        CompiledModel.InputBuffers inputBuffers;
        CompiledModel.OutputBuffers outputBuffers;
        int nIn, nOut;
        long[] inSizes, outSizes;
    }

    // ── lifecycle ─────────────────────────────────────────────
    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if ("STOP".equals(intent != null ? intent.getAction() : null)) {
            stopSelf();
            return START_NOT_STICKY;
        }
        startForeground(CHANNEL_ID_INT, buildNotification("Starting..."));
        startServer();
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        running = false;
        try { if (serverSocket != null) serverSocket.close(); } catch (Exception ignored) {}
        pool.shutdownNow();
        lastStatus = "Stopped";
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent i) { return null; }

    // ── server ────────────────────────────────────────────────
    private void startServer() {
        pool.execute(() -> {
            try {
                serverSocket = new ServerSocket();
                serverSocket.setReuseAddress(true);
                serverSocket.bind(new InetSocketAddress(PORT));
                running = true;
                lastStatus = "Listening on TCP :" + PORT;
                log("Daemon ready on port " + PORT);
                updateNotification(lastStatus);

                while (running) {
                    Socket client = serverSocket.accept();
                    log("Client connected: " + client.getInetAddress());
                    pool.execute(() -> handleClient(client));
                }
            } catch (Exception e) {
                if (running) {
                    log("Server error: " + e.getMessage());
                    lastStatus = "Error: " + e.getMessage();
                }
            }
        });
    }

    // ── client handler ────────────────────────────────────────
    private void handleClient(Socket sock) {
        try {
            sock.setSoTimeout(300_000);
            DataInputStream in = new DataInputStream(new BufferedInputStream(sock.getInputStream()));
            OutputStream out = sock.getOutputStream();

            while (true) {
                int msgType = Integer.reverseBytes(in.readInt());
                int length  = Integer.reverseBytes(in.readInt());

                if (length < 0 || length > 256 * 1024 * 1024) {
                    sendError(out, "payload too large");
                    break;
                }

                byte[] payload = new byte[length];
                in.readFully(payload);
                ByteBuffer buf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);

                switch (msgType) {
                    case MSG_PING:   handlePing(out);        break;
                    case MSG_LOAD:   handleLoad(buf, out);   break;
                    case MSG_INFER:  handleInfer(buf, out);  break;
                    case MSG_BENCH:  handleBench(buf, out);  break;
                    case MSG_UNLOAD: handleUnload(buf, out); break;
                    default:
                        sendError(out, "unknown msg type 0x" + Integer.toHexString(msgType));
                        break;
                }
            }
        } catch (EOFException ignored) {
            log("Client disconnected");
        } catch (Exception e) {
            log("Client error: " + e.getMessage());
        } finally {
            try { sock.close(); } catch (Exception ignored) {}
        }
    }

    // ── PING ──────────────────────────────────────────────────
    private void handlePing(OutputStream out) throws IOException {
        sendMsg(out, MSG_PONG, "PHANTOM_NPU_OK".getBytes());
    }

    // ── UNLOAD ────────────────────────────────────────────────
    private void handleUnload(ByteBuffer buf, OutputStream out) throws IOException {
        int sid = buf.getInt();
        SessionState s = sessions.remove(sid);
        if (s != null) {
            try { s.model.close(); } catch (Exception ignored) {}
            sendAck(out);
            log("Unloaded session " + sid);
        } else {
            sendError(out, "unknown session " + sid);
        }
    }

    // ── LOAD ──────────────────────────────────────────────────
    private void handleLoad(ByteBuffer buf, OutputStream out) throws IOException {
        int nIn  = buf.getInt();
        int nOut = buf.getInt();
        if (nIn <= 0 || nOut <= 0 || nIn > 8 || nOut > 8) {
            sendError(out, "bad tensor counts nIn=" + nIn + " nOut=" + nOut);
            return;
        }
        long[] inSizes  = new long[nIn];
        long[] outSizes = new long[nOut];
        for (int i = 0; i < nIn;  i++) inSizes[i]  = buf.getLong();
        for (int i = 0; i < nOut; i++) outSizes[i] = buf.getLong();
        int blobLen = buf.getInt();
        if (blobLen <= 0 || blobLen > 200 * 1024 * 1024) {
            sendError(out, "bad blob size " + blobLen);
            return;
        }
        byte[] blob = new byte[blobLen];
        buf.get(blob);

        log("Loading model " + blobLen / 1024 + " KB, nIn=" + nIn + " nOut=" + nOut
            + " inSize[0]=" + inSizes[0] + " outSize[0]=" + outSizes[0]);

        File modelFile = null;
        try {
            // Write to cache
            modelFile = new File(getCacheDir(), "model_" + System.currentTimeMillis() + ".tflite");
            try (FileOutputStream fos = new FileOutputStream(modelFile)) { fos.write(blob); }
            log("Model written to: " + modelFile.getAbsolutePath() + " size=" + modelFile.length());

            // Create LiteRT environment and options
            LiteRtEnvironment env = LiteRtEnvironment.create();
            log("LiteRtEnvironment created: " + env);

            LiteRtOptions opts = new LiteRtOptions.Builder()
                .setAccelerator(Accelerator.NPU)
                .build();
            log("LiteRtOptions: NPU accelerator set");

            // Load and compile model
            long loadStart = System.nanoTime();
            CompiledModel model = CompiledModel.create(env, modelFile.getAbsolutePath(), opts);
            long loadMs = (System.nanoTime() - loadStart) / 1_000_000;
            log("CompiledModel.create() took " + loadMs + "ms, model=" + model);

            // Verify model structure
            int actualInputs  = model.getInputCount();
            int actualOutputs = model.getOutputCount();
            log("Model inputs=" + actualInputs + " outputs=" + actualOutputs);
            if (actualInputs == 0 || actualOutputs == 0) {
                sendError(out, "model reports 0 inputs or 0 outputs — compile may have failed");
                return;
            }

            // Create I/O buffers
            CompiledModel.InputBuffers  ib = model.createInputBuffers(0);
            CompiledModel.OutputBuffers ob = model.createOutputBuffers(0);
            log("I/O buffers created: ib=" + ib + " ob=" + ob);

            // Warm-up run to force NPU program upload
            log("Running warm-up inference...");
            long warmStart = System.nanoTime();
            model.run(ib, ob);
            // Full drain: copy all output bytes to force DMA completion
            for (int i = 0; i < nOut; i++) {
                ByteBuffer tensor = ob.get(i);
                tensor.rewind();
                byte[] drain = new byte[tensor.remaining()];
                tensor.get(drain);
            }
            long warmUs = (System.nanoTime() - warmStart) / 1000;
            log("Warm-up done in " + warmUs + "us");

            // Store session
            SessionState s = new SessionState();
            s.id            = nextSessionId++;
            s.model         = model;
            s.inputBuffers  = ib;
            s.outputBuffers = ob;
            s.nIn           = nIn;
            s.nOut          = nOut;
            s.inSizes       = inSizes;
            s.outSizes      = outSizes;
            sessions.put(s.id, s);

            ByteBuffer resp = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            resp.putInt(s.id);
            sendMsg(out, MSG_RESULT, resp.array());
            log("Session " + s.id + " ready");

        } catch (Exception e) {
            log("Load FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            sendError(out, "LOAD failed: " + e.getMessage());
        } finally {
            if (modelFile != null) modelFile.delete();
        }
    }

    // ── INFER ─────────────────────────────────────────────────
    private void handleInfer(ByteBuffer buf, OutputStream out) throws IOException {
        int sid = buf.getInt();
        int ni  = buf.getInt();
        SessionState s = sessions.get(sid);
        if (s == null) { sendError(out, "INFER: unknown session " + sid); return; }

        try {
            // Fill inputs
            for (int i = 0; i < ni; i++) {
                int ilen = buf.getInt();
                byte[] data = new byte[ilen];
                buf.get(data);
                ByteBuffer tensor = s.inputBuffers.get(i);
                tensor.rewind();
                tensor.put(data);
            }

            // Run
            s.model.run(s.inputBuffers, s.outputBuffers);

            // Pack outputs — reading all bytes acts as DMA drain
            int totalSize = 4;
            for (int i = 0; i < s.nOut; i++)
                totalSize += 4 + (int) s.outSizes[i];

            ByteBuffer resp = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);
            resp.putInt(s.nOut);
            for (int i = 0; i < s.nOut; i++) {
                int olen = (int) s.outSizes[i];
                resp.putInt(olen);
                ByteBuffer tensor = s.outputBuffers.get(i);
                tensor.rewind();
                byte[] outData = new byte[olen];
                tensor.get(outData);
                resp.put(outData);
            }
            sendMsg(out, MSG_RESULT, resp.array());

        } catch (Exception e) {
            log("Infer FAILED: " + e.getMessage());
            sendError(out, "INFER failed: " + e.getMessage());
        }
    }

    // ── BENCH ─────────────────────────────────────────────────
    private void handleBench(ByteBuffer buf, OutputStream out) throws IOException {
        int sid  = buf.getInt();
        int runs = buf.getInt();
        if (runs <= 0 || runs > 10000) runs = 50;
        SessionState s = sessions.get(sid);
        if (s == null) { sendError(out, "BENCH: unknown session " + sid); return; }

        try {
            int outBytes = (int) s.outSizes[0];
            // Pre-allocate drain buffer outside loop to avoid GC pressure
            byte[] drain = new byte[outBytes];
            long[] times = new long[runs];

            log("Starting bench: " + runs + " runs, outBytes=" + outBytes);

            for (int r = 0; r < runs; r++) {
                long t0 = System.nanoTime();

                s.model.run(s.inputBuffers, s.outputBuffers);

                // Full output drain: copy all bytes from DMA buffer to heap array.
                // This forces the JVM to wait for every byte the NPU wrote,
                // acting as a real completion barrier.
                ByteBuffer ob = s.outputBuffers.get(0);
                ob.rewind();
                ob.get(drain);  // blocks until all outBytes are readable

                times[r] = (System.nanoTime() - t0) / 1000; // microseconds
            }

            // Compute checksum of last run to verify model is producing output
            int checksum = 0;
            for (byte b : drain) checksum += (b & 0xFF);

            // Stats
            long sum = 0;
            for (long t : times) sum += t;
            long avg = sum / runs;
            long var = 0;
            for (long t : times) var += (t - avg) * (t - avg);
            long std = (long) Math.sqrt((double) var / runs);
            long min = times[0], max = times[0];
            for (long t : times) { if (t < min) min = t; if (t > max) max = t; }

            log("Bench " + runs + " runs:"
                + " avg=" + avg + "us"
                + " std=" + std + "us"
                + " min=" + min + "us"
                + " max=" + max + "us"
                + " checksum=" + checksum);

            ByteBuffer resp = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
            resp.putLong(avg);
            resp.putLong(std);
            sendMsg(out, MSG_RESULT, resp.array());

        } catch (Exception e) {
            log("Bench FAILED: " + e.getMessage());
            sendError(out, "BENCH failed: " + e.getMessage());
        }
    }

    // ── wire helpers ──────────────────────────────────────────
    private void sendMsg(OutputStream out, int type, byte[] payload) throws IOException {
        ByteBuffer hdr = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(type);
        hdr.putInt(payload.length);
        out.write(hdr.array());
        out.write(payload);
        out.flush();
    }

    private void sendError(OutputStream out, String msg) {
        log("Sending error: " + msg);
        try { sendMsg(out, MSG_ERROR, msg.getBytes()); } catch (Exception ignored) {}
    }

    private void sendAck(OutputStream out) throws IOException {
        sendMsg(out, MSG_ACK, new byte[4]);
    }

    // ── notification / logging ────────────────────────────────
    private void createNotificationChannel() {
        NotificationChannel ch = new NotificationChannel(
            CHANNEL_ID, "Phantom NPU Daemon", NotificationManager.IMPORTANCE_LOW);
        getSystemService(NotificationManager.class).createNotificationChannel(ch);
    }

    private Notification buildNotification(String text) {
        return new Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Phantom NPU")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_upload)
            .build();
    }

    private void updateNotification(String text) {
        getSystemService(NotificationManager.class)
            .notify(CHANNEL_ID_INT, buildNotification(text));
    }

    private void log(String msg) {
        Log.i(TAG, msg);
        lastLog = "[" + new java.util.Date() + "] " + msg + "\n" + lastLog;
        if (lastLog.length() > 4000) lastLog = lastLog.substring(0, 4000);
        updateNotification(msg);
    }
}
