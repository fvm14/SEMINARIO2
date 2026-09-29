package pe.edu.ulima.paltascan.datos

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class Inspeccion(
    val id: Long = 0,
    val loteId: Long?,
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
    /** Avisos de ValidacionCaptura separados por coma (OSCURA, BORROSA, BAJA_CONFIANZA). */
    val avisos: String = "",
) {
    val msTotal: Double get() = msPreproceso + msInferencia + msPostproceso
}

data class Lote(
    val id: Long = 0,
    val nombre: String,
    val productor: String,
    val notas: String,
    val fechaMs: Long,
)

/** Conteos de un conjunto de analisis (lote, sesion o historial). */
class Resumen(inspecciones: List<Inspeccion>) {
    val total = inspecciones.size
    val porCategoria = IntArray(3).also { c -> inspecciones.forEach { c[it.categoria]++ } }
    val porMadurez = IntArray(5).also { c -> inspecciones.forEach { c[it.madurez - 1]++ } }
    fun porcentaje(categoria: Int) = if (total == 0) 0 else Math.round(100.0 * porCategoria[categoria] / total).toInt()
}

/** Filtro del historial: por categoria OCDE o por nivel de madurez. */
data class Filtro(val categoria: Int? = null, val madurez: Int? = null)

class BaseDatos(context: Context) : SQLiteOpenHelper(context, "paltascan.db", null, 2) {

    override fun onCreate(db: SQLiteDatabase) {
        crearLotes(db)
        crearInspecciones(db)
    }

    private fun crearLotes(db: SQLiteDatabase) = db.execSQL(
        """
        CREATE TABLE lotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            productor TEXT NOT NULL DEFAULT '',
            notas TEXT NOT NULL DEFAULT '',
            fecha_ms INTEGER NOT NULL
        )
        """.trimIndent()
    )

    private fun crearInspecciones(db: SQLiteDatabase, tabla: String = "inspecciones") {
        db.execSQL(
            """
            CREATE TABLE $tabla (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lote_id INTEGER REFERENCES lotes(id) ON DELETE SET NULL,
                fecha_ms INTEGER NOT NULL,
                archivo_foto TEXT NOT NULL,
                archivo_overlay TEXT NOT NULL,
                madurez INTEGER NOT NULL,
                prob_madurez REAL NOT NULL,
                ratio REAL NOT NULL,
                categoria INTEGER NOT NULL,
                ms_preproceso REAL NOT NULL,
                ms_inferencia REAL NOT NULL,
                ms_postproceso REAL NOT NULL,
                avisos TEXT NOT NULL DEFAULT ''
            )
            """.trimIndent()
        )
        if (tabla == "inspecciones") {
            db.execSQL("CREATE INDEX idx_lote ON inspecciones(lote_id)")
            db.execSQL("CREATE INDEX idx_fecha ON inspecciones(fecha_ms)")
        }
    }

    /** v1 -> v2: el lote era un texto dentro de cada inspeccion; pasa a su propia tabla. */
    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            crearLotes(db)
            db.execSQL(
                "INSERT INTO lotes (nombre, fecha_ms) SELECT lote, MIN(fecha_ms) FROM inspecciones GROUP BY lote"
            )
            crearInspecciones(db, "inspecciones_v2")
            db.execSQL(
                """
                INSERT INTO inspecciones_v2 (id, lote_id, fecha_ms, archivo_foto, archivo_overlay, madurez,
                    prob_madurez, ratio, categoria, ms_preproceso, ms_inferencia, ms_postproceso)
                SELECT i.id, l.id, i.fecha_ms, i.archivo_foto, i.archivo_overlay, i.madurez, i.prob_madurez,
                    i.ratio, i.categoria, i.ms_preproceso, i.ms_inferencia, i.ms_postproceso
                FROM inspecciones i JOIN lotes l ON l.nombre = i.lote
                """.trimIndent()
            )
            db.execSQL("DROP TABLE inspecciones")
            db.execSQL("ALTER TABLE inspecciones_v2 RENAME TO inspecciones")
            db.execSQL("CREATE INDEX idx_lote ON inspecciones(lote_id)")
            db.execSQL("CREATE INDEX idx_fecha ON inspecciones(fecha_ms)")
        }
    }

    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
    }

    // ---------------------------------------------------------------- inspecciones

    fun insertar(i: Inspeccion): Long = writableDatabase.insert("inspecciones", null, ContentValues().apply {
        put("lote_id", i.loteId)
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
        put("avisos", i.avisos)
    })

    fun obtener(id: Long): Inspeccion? =
        consultar("SELECT * FROM inspecciones WHERE id = ?", arrayOf(id.toString())).firstOrNull()

    fun historial(filtro: Filtro = Filtro(), limite: Int? = null): List<Inspeccion> {
        val donde = mutableListOf<String>()
        val args = mutableListOf<String>()
        filtro.categoria?.let { donde += "categoria = ?"; args += it.toString() }
        filtro.madurez?.let { donde += "madurez = ?"; args += it.toString() }
        val sql = "SELECT * FROM inspecciones" +
            (if (donde.isEmpty()) "" else " WHERE " + donde.joinToString(" AND ")) +
            " ORDER BY fecha_ms DESC" + (limite?.let { " LIMIT $it" } ?: "")
        return consultar(sql, args.toTypedArray())
    }

    fun porLote(loteId: Long): List<Inspeccion> =
        consultar("SELECT * FROM inspecciones WHERE lote_id = ? ORDER BY fecha_ms DESC", arrayOf(loteId.toString()))

    fun contarInspecciones(): Int =
        readableDatabase.rawQuery("SELECT COUNT(*) FROM inspecciones", null).use { it.moveToFirst(); it.getInt(0) }

    fun moverALote(inspeccionId: Long, loteId: Long?) {
        writableDatabase.update("inspecciones", ContentValues().apply { put("lote_id", loteId) },
            "id = ?", arrayOf(inspeccionId.toString()))
    }

    fun eliminar(id: Long) = writableDatabase.delete("inspecciones", "id = ?", arrayOf(id.toString()))

    private fun consultar(sql: String, args: Array<String>): List<Inspeccion> =
        readableDatabase.rawQuery(sql, args).use { c -> buildList { while (c.moveToNext()) add(leer(c)) } }

    private fun leer(c: Cursor) = Inspeccion(
        id = c.getLong(c.getColumnIndexOrThrow("id")),
        loteId = c.getColumnIndexOrThrow("lote_id").let { if (c.isNull(it)) null else c.getLong(it) },
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
        avisos = c.getString(c.getColumnIndexOrThrow("avisos")),
    )

    // ---------------------------------------------------------------- lotes

    fun crearLote(l: Lote): Long = writableDatabase.insert("lotes", null, ContentValues().apply {
        put("nombre", l.nombre)
        put("productor", l.productor)
        put("notas", l.notas)
        put("fecha_ms", l.fechaMs)
    })

    fun lote(id: Long): Lote? =
        consultarLotes("SELECT * FROM lotes WHERE id = ?", arrayOf(id.toString())).firstOrNull()

    fun lotes(): List<Lote> = consultarLotes("SELECT * FROM lotes ORDER BY fecha_ms DESC", emptyArray())

    fun contarLotes(): Int =
        readableDatabase.rawQuery("SELECT COUNT(*) FROM lotes", null).use { it.moveToFirst(); it.getInt(0) }

    private fun consultarLotes(sql: String, args: Array<String>): List<Lote> =
        readableDatabase.rawQuery(sql, args).use { c ->
            buildList {
                while (c.moveToNext()) add(
                    Lote(
                        id = c.getLong(c.getColumnIndexOrThrow("id")),
                        nombre = c.getString(c.getColumnIndexOrThrow("nombre")),
                        productor = c.getString(c.getColumnIndexOrThrow("productor")),
                        notas = c.getString(c.getColumnIndexOrThrow("notas")),
                        fechaMs = c.getLong(c.getColumnIndexOrThrow("fecha_ms")),
                    )
                )
            }
        }
}
