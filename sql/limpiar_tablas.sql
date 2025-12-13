CREATE PROCEDURE sp_limpiar_tablas
AS
BEGIN
    DELETE FROM fact_historia_clinica;
    DELETE FROM dim_especialidad;
    DELETE FROM dim_diagnostico;
    DELETE FROM dim_tratamiento;
    DELETE FROM staging_historias;
    
    DBCC CHECKIDENT ('fact_historia_clinica', RESEED, 0);
    DBCC CHECKIDENT ('dim_especialidad', RESEED, 0);
    DBCC CHECKIDENT ('dim_diagnostico', RESEED, 0);
    DBCC CHECKIDENT ('dim_tratamiento', RESEED, 0);
END
