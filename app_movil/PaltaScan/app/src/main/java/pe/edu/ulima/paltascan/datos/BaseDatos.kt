package pe.edu.ulima.paltascan.datos

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class Inspeccion(
    val id: Long = 0,
    val lote: String,
    val fechaMs: Long,
    val archivoFoto: String,
    val archivoOverlay: String,
    val madurez: Int,
    val probMadurez: Float,
    val ratio: Double,
    val categoria: Int,
    val msPreproceso: Double,
    val msInferencia: Double,
    val msPostproceso: Double,
) {
    val msTotal: Double get() = msPreproceso + msInferencia + msPostproceso
}

class BaseDatos(context: Context) : SQLiteOpenHelper(context, "paltascan.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE inspecciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lote TEXT NOT NULL,
                fecha_ms INTEGER NOT NULL,
                archivo_foto TEXT NOT NULL,
                archivo_overlay TEXT NOT NULL,
                madurez INTEGER NOT NULL,
                prob_madurez REAL NOT NULL,
                ratio REAL NOT NULL,
                categoria INTEGER NOT NULL,
                ms_preproceso REAL NOT NULL,
                ms_inferencia REAL NOT NULL,
                ms_postproceso REAL NOT NULL
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX idx_lote ON inspecciones(lote)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit

    fun insertar(i: Inspeccion): Long = writableDatabase.insert("inspecciones", null, ContentValues().apply {
        put("lote", i.lote)
        put("fecha_ms", i.fechaMs)
        put("archivo_foto", i.archivoFoto)
        put("archivo_overlay", i.archivoOverlay)
        put("madurez", i.madurez)
        put("prob_madurez", i.probMadurez)
        put("ratio", i.ratio)
        put("categoria", i.categoria)
        put("ms_preproceso", i.msPreproceso)
        put("ms_inferencia", i.msInferencia)
        put("ms_postproceso", i.msPostproceso)
    })

    fun obtener(id: Long): Inspeccion? =
        readableDatabase.rawQuery("SELECT * FROM inspecciones WHERE id = ?", arrayOf(id.toString())).use { c ->
            if (c.moveToFirst()) leer(c) else null
        }

    fun porLote(lote: String): List<Inspeccion> =
        readableDatabase.rawQuery("SELECT * FROM inspecciones WHERE lote = ? ORDER BY fecha_ms DESC", arrayOf(lote)).use { c ->
            buildList { while (c.moveToNext()) add(leer(c)) }
        }

    fun lotes(): List<String> =
        readableDatabase.rawQuery("SELECT lote FROM inspecciones GROUP BY lote ORDER BY MAX(fecha_ms) DESC", null).use { c ->
            buildList { while (c.moveToNext()) add(c.getString(0)) }
        }

    fun eliminar(id: Long) = writableDatabase.delete("inspecciones", "id = ?", arrayOf(id.toString()))

    private fun leer(c: Cursor) = Inspeccion(
        id = c.getLong(c.getColumnIndexOrThrow("id")),
        lote = c.getString(c.getColumnIndexOrThrow("lote")),
        fechaMs = c.getLong(c.getColumnIndexOrThrow("fecha_ms")),
        archivoFoto = c.getString(c.getColumnIndexOrThrow("archivo_foto")),
        archivoOverlay = c.getString(c.getColumnIndexOrThrow("archivo_overlay")),
        madurez = c.getInt(c.getColumnIndexOrThrow("madurez")),
        probMadurez = c.getFloat(c.getColumnIndexOrThrow("prob_madurez")),
        ratio = c.getDouble(c.getColumnIndexOrThrow("ratio")),
        categoria = c.getInt(c.getColumnIndexOrThrow("categoria")),
        msPreproceso = c.getDouble(c.getColumnIndexOrThrow("ms_preproceso")),
        msInferencia = c.getDouble(c.getColumnIndexOrThrow("ms_inferencia")),
        msPostproceso = c.getDouble(c.getColumnIndexOrThrow("ms_postproceso")),
    )
}
